#pragma once

#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <k4a/k4a.hpp>
#include <glm/glm.hpp>

// BGRA to BGR
cv::Mat color_to_opencv(const k4a::image& i);

cv::Mat color_to_opencv_gray(const k4a::image& i);

cv::Mat depth_to_opencv(const k4a::image& i);

cv::Mat point_cloud_to_opencv(const k4a::image& i);

cv::Vec4d position_to_opencv_vec4d(const k4a_float3_t& position);

float distance(const glm::vec3& p1, const glm::vec3& p2);

Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
point_cloud_to_eigen(k4a::image point_cloud_image, int num_rows);

struct TransformMatrix {
	cv::Matx33d R;
	cv::Vec3d t;
	cv::Matx44d homogeneous_matrix;

	TransformMatrix();

	// construct from a homogeneous matrix
	TransformMatrix(const cv::Matx44d& H);

	TransformMatrix(const cv::Matx33d R, const cv::Vec3d t);

	// build from k4a transformation
	TransformMatrix(const k4a::calibration cal, k4a_calibration_type_t from, k4a_calibration_type_t to);

	// read from file
	TransformMatrix(std::string file_name);

	cv::Matx44d to_homogeneous() const;

	TransformMatrix compose(const TransformMatrix second_tm) const;

	void store_to_file(std::string file_name) const;

	Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> point_cloud_transformation(
		const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& input
	);

	glm::mat4 to_glm_mat4() const;

	cv::Vec4d dot(const cv::Vec4d& vector) const;
};
