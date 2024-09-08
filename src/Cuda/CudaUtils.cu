#include "../../include/Cuda/CudaUtils.cuh"

#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include "../../include/PointCloud.h"

struct BGRA2RGB {
	__host__ __device__ My::ColorRGB operator()(const My::ColorBGRA bgra) {
		return My::ColorRGB{ bgra.r, bgra.g, bgra.b };
	}
};

struct BGRA2BGR {
	__host__ __device__ My::ColorBGR operator()(const My::ColorBGRA bgra) {
		return My::ColorBGR{ bgra.b, bgra.g, bgra.r };
	}
};

void cuda_BGRA2RGB(std::vector<My::ColorBGRA>& input, std::vector<My::ColorRGB>& output)
{
	using namespace thrust;
	using My::ColorRGB;
	using My::ColorBGRA;

	device_vector<ColorBGRA> d_colorBGRA = input;
	device_vector<ColorRGB> d_colorRGB(input.size());
	transform(
		d_colorBGRA.begin(),
		d_colorBGRA.end(),
		d_colorRGB.begin(),
		BGRA2RGB()
	);
	output.resize(input.size());
	thrust::copy(
		d_colorRGB.begin(),
		d_colorRGB.end(),
		output.begin()
	);
}

void cuda_BGRA2BGR(std::vector<My::ColorBGRA>& input, std::vector<My::ColorBGR>& output)
{
	using namespace thrust;
	using My::ColorBGRA;
	using My::ColorBGR;

	device_vector<ColorBGRA> d_colorBGRA = input;
	device_vector<ColorBGR> d_colorBGR(input.size());
	transform(
		d_colorBGRA.begin(),
		d_colorBGRA.end(),
		d_colorBGR.begin(),
		BGRA2BGR()
	);
	output.resize(input.size());
	thrust::copy(
		d_colorBGR.begin(),
		d_colorBGR.end(),
		output.begin()
	);
}

struct ZipPointAndColorBGRAToPointColorRGB {
	__host__ __device__ My::PointColorRGB operator()(const thrust::tuple<My::Point, My::ColorBGRA> t) {
		My::ColorBGRA bgra = thrust::get<1>(t);
		return My::PointColorRGB{
			thrust::get<0>(t),
			My::ColorRGB{bgra.r, bgra.g, bgra.b}
		};
	}
};

void cuda_zip_point_color(
	const std::vector<My::Point>& point_input,
	const std::vector<My::ColorBGRA>& color_input,
	std::vector<My::PointColorRGB>& output
) {
	using namespace thrust;
	using My::Point;
	using My::ColorBGRA;
	using My::ColorRGB;

	device_vector<My::Point> d_point = point_input;
	device_vector<My::ColorBGRA> d_colorBGRA = color_input;
	device_vector<My::PointColorRGB> d_output(point_input.size());

	transform(
		make_zip_iterator(d_point.begin(), d_colorBGRA.begin()),
		make_zip_iterator(d_point.end(), d_colorBGRA.end()),
		d_output.begin(),
		ZipPointAndColorBGRAToPointColorRGB()
	);
	output.resize(point_input.size());
	thrust::copy(
		d_output.begin(),
		d_output.end(),
		output.begin()
	);
}

struct UnzipPointColorRGBToPointAndColorBGRA {
	__host__ __device__ thrust::tuple<My::Point, My::ColorBGRA> operator()(const My::PointColorRGB input) {
		My::Point p = input.pos;
		My::ColorBGRA colorBGRA{ input.color.b, input.color.g, input.color.r, 255 };
		return thrust::make_tuple(p, colorBGRA);
	}
};

void cuda_unzip_point_color(
	const std::vector<My::PointColorRGB>& point_colorRGB_input,
	std::vector<My::Point>& point_output,
	std::vector<My::ColorBGRA>& colorBGRA_output
) {
	using namespace thrust;
	using My::PointColorRGB;
	using My::Point;
	using My::ColorBGRA;

	device_vector<Point> d_output_point(point_colorRGB_input.size());
	device_vector<ColorBGRA> d_output_colorBGRA(point_colorRGB_input.size());
	device_vector<PointColorRGB> d_input_point_colorRGB = point_colorRGB_input;

	transform(
		d_input_point_colorRGB.begin(),
		d_input_point_colorRGB.end(),
		make_zip_iterator(d_output_point.begin(), d_output_colorBGRA.begin()),
		UnzipPointColorRGBToPointAndColorBGRA()
	);
	point_output.resize(d_output_point.size());
	colorBGRA_output.resize(d_output_colorBGRA.size());
	thrust::copy(d_output_point.begin(), d_output_point.end(), point_output.begin());
	thrust::copy(d_output_colorBGRA.begin(), d_output_colorBGRA.end(), colorBGRA_output.begin());
}

struct UnzipPointColorRGBToPointAndColorRGB {
	__host__ __device__ thrust::tuple<My::Point, My::ColorRGB> operator()(const My::PointColorRGB input) {
		My::Point p = input.pos;
		My::ColorRGB colorBGRA{ input.color.r, input.color.g, input.color.b };
		return thrust::make_tuple(p, colorBGRA);
	}
};

void cuda_unzip_point_color(const std::vector<My::PointColorRGB>& point_colorRGB_input, std::vector<My::Point>& point_output, std::vector<My::ColorRGB>& colorRGB_output) 
{
	using namespace thrust;
	using My::PointColorRGB;
	using My::Point;
	using My::ColorRGB;

	device_vector<Point> d_output_point(point_colorRGB_input.size());
	device_vector<ColorRGB> d_output_colorRGB(point_colorRGB_input.size());
	device_vector<PointColorRGB> d_input_point_colorRGB = point_colorRGB_input;

	transform(
		d_input_point_colorRGB.begin(),
		d_input_point_colorRGB.end(),
		make_zip_iterator(d_output_point.begin(), d_output_colorRGB.begin()),
		UnzipPointColorRGBToPointAndColorRGB()
	);
	point_output.resize(d_output_point.size());
	colorRGB_output.resize(d_output_colorRGB.size());
	thrust::copy(d_output_point.begin(), d_output_point.end(), point_output.begin());
	thrust::copy(d_output_colorRGB.begin(), d_output_colorRGB.end(), colorRGB_output.begin());
}