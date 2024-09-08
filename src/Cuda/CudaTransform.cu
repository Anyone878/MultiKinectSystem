#include "../../include/Cuda/CudaTransform.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <vector_functions.hpp>

#include "../../include/Cuda/CudaStreamManager.cuh"
#include "../../include/Cuda/CublasHandleManager.cuh"

struct ConvertToHomogeneous {
	__host__ __device__ float4 operator()(const My::Point& p) const {
		return make_float4(
			static_cast<float>(p.x),
			static_cast<float>(p.y),
			static_cast<float>(p.z),
			1.0f);
	}
};

struct ConvertToCartesian {
	__host__ __device__ My::Point operator()(const float4& p) const {
		using My::Point;
		//float inv_w = (p.w != 0.0f) ? 1.0f / p.w : 0.0f;
		return Point{
			static_cast<int16_t>(p.x/* * inv_w*/),
			static_cast<int16_t>(p.y/* * inv_w*/),
			static_cast<int16_t>(p.z/* * inv_w*/)
		};
	}
};

My::PointCloud cuda_transformation(const My::PointCloud& point_cloud, const cv::Matx44d& T, const int processing_index)
{
	using namespace My;
	size_t num_points = point_cloud.points.size();
	thrust::device_vector<Point> d_points(point_cloud.points);
	thrust::device_vector<float4> d_homogeneous_points(num_points);
	// convert to homogeneous
	thrust::transform(
		d_points.begin(),
		d_points.end(),
		d_homogeneous_points.begin(),
		ConvertToHomogeneous()
	);

	// prepare transform matrix
	thrust::host_vector<float> h_T(16);
	for (int i = 0; i < 16; ++i) {
		h_T[i] = static_cast<float>(T.val[i]);
	}
	thrust::device_vector<float> d_T = h_T;

	// set handle and stream
	cublasHandle_t handle = CublasHandleManager::get_instance().get_handle(processing_index);
	cudaStream_t stream = CudaStreamManager::get_instance().get_stream(processing_index);
	cublasSetStream(handle, stream);

	thrust::device_vector<float4> d_transformed_points(num_points);
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasSgemm(
		handle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		4, num_points, 4,
		&alpha,
		thrust::raw_pointer_cast(d_T.data()), 4,
		reinterpret_cast<float*>(thrust::raw_pointer_cast(d_homogeneous_points.data())), 4,
		&beta,
		reinterpret_cast<float*>(thrust::raw_pointer_cast(d_transformed_points.data())), 4
	);

	// convert results to cartesian points
	thrust::device_vector<Point> d_cartesian_points(num_points);
	thrust::transform(
		d_transformed_points.begin(),
		d_transformed_points.end(),
		d_cartesian_points.begin(),
		ConvertToCartesian()
	);

	// copy to host vector
	PointCloud output;
	output.points.resize(num_points);
	thrust::copy(d_cartesian_points.begin(), d_cartesian_points.end(), output.points.begin());
	output.colors.resize(num_points);
	thrust::copy(point_cloud.colors.begin(), point_cloud.colors.end(), output.colors.begin());
	return output;
}