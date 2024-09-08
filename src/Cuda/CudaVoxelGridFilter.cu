#include "../../include/Cuda/CudaVoxelGridFilter.cuh"

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

#include "../../include/PointCloud.h"


__global__ void _kernel_compute_voxel_grid_index(
	const My::PointColorRGB* point_colorRGB,
	int* voxel_indices,
	int num_points,
	float voxel_size,
	int3 grid_size
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < num_points) {
		int x = floorf(point_colorRGB[index].pos.x / voxel_size);
		int y = floorf(point_colorRGB[index].pos.y / voxel_size);
		int z = floorf(point_colorRGB[index].pos.z / voxel_size);
		voxel_indices[index] = x + y * grid_size.x + z * grid_size.x * grid_size.y;
	}
}

struct PointColorRGBPro {
	__host__ __device__ PointColorRGBPro(const My::PointColorRGB p)
		: x(static_cast<int>(p.pos.x))
		, y(static_cast<int>(p.pos.y))
		, z(static_cast<int>(p.pos.z))
		, r(static_cast<int>(p.color.r))
		, g(static_cast<int>(p.color.g))
		, b(static_cast<int>(p.color.b)) {}

	__host__ __device__ PointColorRGBPro(const My::Point p, const My::ColorRGB c)
		: x(static_cast<int>(p.x))
		, y(static_cast<int>(p.y))
		, z(static_cast<int>(p.z))
		, r(static_cast<int>(c.r))
		, g(static_cast<int>(c.g))
		, b(static_cast<int>(c.b))
	{}

	__host__ __device__ PointColorRGBPro(const My::Point p, const My::ColorBGRA c)
		: x(static_cast<int>(p.x))
		, y(static_cast<int>(p.y))
		, z(static_cast<int>(p.z))
		, r(static_cast<int>(c.r))
		, g(static_cast<int>(c.g))
		, b(static_cast<int>(c.b))
	{}

	__host__ __device__ PointColorRGBPro(const int x, const int y, const int z, const int r, const int g, const int b)
		: x(x), y(y), z(z), r(r), g(g), b(b) {}

	__host__ __device__ PointColorRGBPro() : x(0), y(0), z(0), r(0), g(0), b(0) {}

	int x, y, z, r, g, b;

	__host__ __device__ My::PointColorRGB to_point_colorRGB() {
		return My::PointColorRGB{
			static_cast<int16_t>(x),
			static_cast<int16_t>(y),
			static_cast<int16_t>(z),
			static_cast<uint8_t>(r),
			static_cast<uint8_t>(g),
			static_cast<uint8_t>(b)
		};
	}
};

struct ZipPointAndColorBGRAToPointColorRGBPro {
	__host__ __device__ PointColorRGBPro operator()(const thrust::tuple<My::Point, My::ColorBGRA> t) {
		return PointColorRGBPro(thrust::get<0>(t), thrust::get<1>(t));
	}
};


struct ComparePosX {
	__host__ __device__ bool operator()(const PointColorRGBPro left, const PointColorRGBPro right) {
		return left.x < right.x;
	}
};

struct ComparePosY {
	__host__ __device__ bool operator()(const PointColorRGBPro left, const PointColorRGBPro right) {
		return left.y < right.y;
	}
};

struct ComparePosZ {
	__host__ __device__ bool operator()(const PointColorRGBPro left, const PointColorRGBPro right) {
		return left.z < right.z;
	}
};

struct FindVoxelIndex {
	float voxel_size;
	int3 grid_size;

	FindVoxelIndex(float voxel_size, int3 grid_size) : voxel_size(voxel_size), grid_size(grid_size) {}

	__host__ __device__ int operator()(const PointColorRGBPro p) {
		int x = floorf(p.x / voxel_size);
		int y = floorf(p.y / voxel_size);
		int z = floorf(p.z / voxel_size);
		return x + y * grid_size.x + z * grid_size.x * grid_size.y;
	}
};

struct AccumulatePointColorRGB {
	__host__ __device__ PointColorRGBPro operator()(const PointColorRGBPro a, const PointColorRGBPro b) {
		return PointColorRGBPro(
			a.x + b.x,
			a.y + b.y,
			a.z + b.z,
			a.r + b.r,
			a.g + b.g,
			a.b + b.b
		);
	}
};

struct AveragePointColorRGBProValueToPointColorRGB {
	__host__ __device__ My::PointColorRGB operator()(thrust::tuple<PointColorRGBPro, int> t) {
		PointColorRGBPro p = thrust::get<0>(t);
		int count = thrust::get<1>(t);
		return My::PointColorRGB{
			static_cast<int16_t>(p.x / count),
			static_cast<int16_t>(p.y / count),
			static_cast<int16_t>(p.z / count),
			static_cast<uint8_t>(p.r / count),
			static_cast<uint8_t>(p.g / count),
			static_cast<uint8_t>(p.b / count),
		};
	}
};

void cuda_voxel_grid_filter(
	const My::PointCloud& input_point_cloud,
	const std::vector<My::ColorBGRA>& input_colorBGRA,
	const float voxel_size,
	std::vector<My::PointColorRGB>& output)
{
	using namespace thrust;
	using My::Point;
	using My::ColorBGRA;
	using My::ColorRGB;
	using My::PointColorRGB;
	using My::PointCloud;

	// zip point and color to PointColorRGB
	device_vector<Point> d_input_point = input_point_cloud.points;
	device_vector<ColorBGRA> d_input_colorBGRA = input_colorBGRA;
	device_vector<PointColorRGBPro> d_point_colorRGBPro(input_colorBGRA.size());
	transform(
		make_zip_iterator(d_input_point.begin(), d_input_colorBGRA.begin()),
		make_zip_iterator(d_input_point.end(), d_input_colorBGRA.end()),
		d_point_colorRGBPro.begin(),
		ZipPointAndColorBGRAToPointColorRGBPro()
	);

	// compute grid size
	device_vector<PointColorRGBPro>::iterator it_maxX = max_element(d_point_colorRGBPro.begin(), d_point_colorRGBPro.end(), ComparePosX());
	device_vector<PointColorRGBPro>::iterator it_maxY = max_element(d_point_colorRGBPro.begin(), d_point_colorRGBPro.end(), ComparePosY());
	device_vector<PointColorRGBPro>::iterator it_maxZ = max_element(d_point_colorRGBPro.begin(), d_point_colorRGBPro.end(), ComparePosZ());
	int positionX = it_maxX - d_point_colorRGBPro.begin();
	int positionY = it_maxY - d_point_colorRGBPro.begin();
	int positionZ = it_maxZ - d_point_colorRGBPro.begin();
	PointColorRGBPro X = d_point_colorRGBPro[positionX];
	PointColorRGBPro Y = d_point_colorRGBPro[positionY];
	PointColorRGBPro Z = d_point_colorRGBPro[positionZ];
	int3 grid_size{
		static_cast<int>(ceil(X.x / voxel_size)),
		static_cast<int>(ceil(Y.y / voxel_size)),
		static_cast<int>(ceil(Z.z / voxel_size))
	};

	// find the voxel_index of each point
	device_vector<int> d_voxel_indices(input_colorBGRA.size());
	transform(
		d_point_colorRGBPro.begin(),
		d_point_colorRGBPro.end(),
		d_voxel_indices.begin(),
		FindVoxelIndex(voxel_size, grid_size)
	);

	// compute the centroid
	sort_by_key(d_voxel_indices.begin(), d_voxel_indices.end(), d_point_colorRGBPro.begin());

	device_vector<int> d_reduced_voxel_indices(d_voxel_indices.size());
	device_vector<PointColorRGBPro> d_reduced_point_colorRGBPro(d_point_colorRGBPro.size());
	thrust::equal_to<int> binary_pred_equal;
	thrust::plus<int> binary_op_plus;
	thrust::pair<device_vector<int>::iterator, device_vector<PointColorRGBPro>::iterator> new_end = reduce_by_key(
		d_voxel_indices.begin(),
		d_voxel_indices.end(),
		d_point_colorRGBPro.begin(),
		d_reduced_voxel_indices.begin(),
		d_reduced_point_colorRGBPro.begin(),
		binary_pred_equal,
		AccumulatePointColorRGB()
	);
	int new_size = new_end.first - d_reduced_voxel_indices.begin();
	d_reduced_voxel_indices.resize(new_size);
	d_reduced_point_colorRGBPro.resize(new_size);

	device_vector<int> d_one_helper_reduced_voxel_indices(d_point_colorRGBPro.size());
	device_vector<int> d_one_helper_reduced_result(d_point_colorRGBPro.size());
	thrust::pair<device_vector<int>::iterator, device_vector<int>::iterator> one_new_end = reduce_by_key(
		d_voxel_indices.begin(),
		d_voxel_indices.end(),
		make_constant_iterator(1),
		d_one_helper_reduced_voxel_indices.begin(),
		d_one_helper_reduced_result.begin(),
		binary_pred_equal,
		binary_op_plus
	);
	int new_size_one_helper = one_new_end.first - d_one_helper_reduced_voxel_indices.begin();
	d_one_helper_reduced_result.resize(new_size_one_helper);

	// finding the centroid for each voxel
	device_vector<My::PointColorRGB> d_output(new_size);
	transform(
		make_zip_iterator(d_reduced_point_colorRGBPro.begin(), d_one_helper_reduced_result.begin()),
		make_zip_iterator(d_reduced_point_colorRGBPro.end(), d_one_helper_reduced_result.end()),
		d_output.begin(),
		AveragePointColorRGBProValueToPointColorRGB()
	);
	output.resize(new_size);
	thrust::copy(
		d_output.begin(),
		d_output.end(),
		output.begin()
	);
}
