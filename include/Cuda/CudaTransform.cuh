#ifndef CUDA_TRANSFORM_H
#define CUDA_TRANSFORM_H

#include <cublas_v2.h>

#include <opencv2/opencv.hpp>
#include "../PointCloud.h"

My::PointCloud cuda_transformation(const My::PointCloud& point_cloud, const cv::Matx44d& T, const int processing_index);

#endif // CUDA_TRANSFORM_H
