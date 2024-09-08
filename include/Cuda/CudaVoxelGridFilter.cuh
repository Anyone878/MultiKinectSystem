#ifndef CUDA_VOXEL_GRID_FILTER_CUH
#define CUDA_VOXEL_GRID_FILTER_CUH

#include <vector>

#include "../PointCloud.h"

void cuda_voxel_grid_filter(
	const My::PointCloud& input_point_cloud, 
	const std::vector<My::ColorBGRA>& input_colorBGRA, 
	const float voxel_size,
	std::vector<My::PointColorRGB>& output
);

void cuda_voxel_grid_filter(
	const std::vector<My::PointCloud>& input_point_cloud,
	const float voxel_size,
	std::vector<My::PointColorRGB>& output
);

#endif // !CUDA_VOXEL_GRID_FILTER_CUH
