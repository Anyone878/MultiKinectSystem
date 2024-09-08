#include "../../include/Utils/MatrixUtils.h"

using namespace cv;
using namespace k4a;

// BGRA to BGR
Mat color_to_opencv(const image& i) {
	Mat cv_image_with_alpha(i.get_height_pixels(), i.get_width_pixels(), CV_8UC4, (void*)i.get_buffer());
	Mat cv_image_no_alpha;
	cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, COLOR_BGRA2BGR);
	return cv_image_no_alpha;
}

Mat color_to_opencv_gray(const image& i) {
	Mat cv_image_with_alpha(i.get_height_pixels(), i.get_width_pixels(), CV_8UC4, (void*)i.get_buffer());
	Mat cv_image_gray;
	cv::cvtColor(cv_image_with_alpha, cv_image_gray, COLOR_BGRA2GRAY);
	return cv_image_gray;
}

Mat depth_to_opencv(const image& i) {
	return Mat(
		i.get_height_pixels(),
		i.get_width_pixels(),
		CV_16U,
		(void*)i.get_buffer(),
		static_cast<size_t>(i.get_stride_bytes())
	);
}

Mat point_cloud_to_opencv(const image& i) {
	return Mat(
		i.get_height_pixels() * i.get_width_pixels(),
		3,
		CV_16S,
		(void*)i.get_buffer()
	);
}

Vec4d position_to_opencv_vec4d(const k4a_float3_t& position) {
	return Vec4d(position.xyz.x, position.xyz.y, position.xyz.z, 1.);
}

float distance(const glm::vec3& p1, const glm::vec3& p2) {
	return glm::distance(p1, p2);
}

Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> point_cloud_to_eigen(image point_cloud_image, int num_rows) {
	Eigen::Map<const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		mapped_point_cloud((int16_t*)(void*)point_cloud_image.get_buffer(), num_rows, 3);

	return mapped_point_cloud;
}

TransformMatrix::TransformMatrix() : R(Matx33d::eye()), t(0., 0., 0.), homogeneous_matrix(Matx44d::eye()) {}

// construct from a homogeneous matrix

TransformMatrix::TransformMatrix(const Matx44d& H) : R(H.get_minor<3, 3>(0, 0)), t(H(0, 3), H(1, 3), H(2, 3)) {
	homogeneous_matrix = Matx44d(
		R(0, 0), R(0, 1), R(0, 2), t(0),
		R(1, 0), R(1, 1), R(1, 2), t(1),
		R(2, 0), R(2, 1), R(2, 2), t(2),
		0, 0, 0, 1
	);
}

TransformMatrix::TransformMatrix(const Matx33d R, const Vec3d t) : R(R), t(t) {
	homogeneous_matrix = Matx44d(
		R(0, 0), R(0, 1), R(0, 2), t(0),
		R(1, 0), R(1, 1), R(1, 2), t(1),
		R(2, 0), R(2, 1), R(2, 2), t(2),
		0, 0, 0, 1
	);
}

TransformMatrix::TransformMatrix(const k4a::calibration cal, k4a_calibration_type_t from, k4a_calibration_type_t to)
{
	const k4a_calibration_extrinsics_t ex = cal.extrinsics[from][to];
	R = Matx33d(
		ex.rotation[0], ex.rotation[1], ex.rotation[2],
		ex.rotation[3], ex.rotation[4], ex.rotation[5],
		ex.rotation[6], ex.rotation[7], ex.rotation[8]
	);
	t = Vec3d(ex.translation[0], ex.translation[1], ex.translation[2]);
	homogeneous_matrix = Matx44d(
		R(0, 0), R(0, 1), R(0, 2), t(0),
		R(1, 0), R(1, 1), R(1, 2), t(1),
		R(2, 0), R(2, 1), R(2, 2), t(2),
		0, 0, 0, 1
	);
}

// read from file
TransformMatrix::TransformMatrix(std::string file_name) {
	FileStorage fs(file_name, FileStorage::READ | FileStorage::FORMAT_JSON);
	fs["R"] >> R;
	fs["t"] >> t;
	fs.release();

	homogeneous_matrix = Matx44d(
		R(0, 0), R(0, 1), R(0, 2), t(0),
		R(1, 0), R(1, 1), R(1, 2), t(1),
		R(2, 0), R(2, 1), R(2, 2), t(2),
		0, 0, 0, 1
	);
}

Matx44d TransformMatrix::to_homogeneous() const {
	return homogeneous_matrix;
}

TransformMatrix TransformMatrix::compose(const TransformMatrix second_tm) const {
	Matx44d H1 = to_homogeneous();
	Matx44d H2 = second_tm.to_homogeneous();
	Matx44d H3 = H1 * H2;
	return TransformMatrix(H3);
}

void TransformMatrix::store_to_file(std::string file_name) const {
	FileStorage fs(file_name, FileStorage::WRITE | FileStorage::FORMAT_JSON);
	fs << "R" << R;
	fs << "t" << t;
	fs.release();
}

Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TransformMatrix::point_cloud_transformation(const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& input) {
	// Rotation and translation matrix to eigen
	Eigen::Matrix3d R_eigen;
	Eigen::Vector3d t_eigen;
	cv2eigen(R, R_eigen);
	cv2eigen(t, t_eigen);
	// cast to double
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> input_double = input.cast<double>();
	// calculation
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output_double =
		(input_double * R_eigen.transpose()).rowwise() + t_eigen.transpose();
	// cast to int16_t
	return output_double.cast<int16_t>();
}

glm::mat4 TransformMatrix::to_glm_mat4() const {
	glm::mat4 t;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			t[j][i] = static_cast<float>(homogeneous_matrix(i, j));
		}
	}
	return t;
}

Vec4d TransformMatrix::dot(const Vec4d& vector) const {
	return homogeneous_matrix * vector;
}
