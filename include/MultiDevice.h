#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <spdlog/spdlog.h>
#include <k4a/k4a.hpp>
#include <fmt/core.h>

#include "./Utils/MatrixUtils.h"

using namespace k4a;
using namespace std;

constexpr chrono::microseconds MAX_ALLOWABLE_TIME_OFFSET_FOR_IMAGE_TIMESTAMP(100);
constexpr int64_t WAIT_FOR_SYNC_CAPTURE_TIMEOUT = 60000;
constexpr uint32_t MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_US = 160;

static void log_lagging_time(string lagger, capture& master, capture sub) {
	spdlog::info("{:<8} Master Camera: {}us; Sub Camera: {}us.",
		lagger,
		master.get_color_image().get_device_timestamp().count(),
		sub.get_color_image().get_device_timestamp().count());
}

static void log_sync_image_time(capture& master, capture& sub) {
	spdlog::info("Sync'd caputre Master Camera: {}us; Sub Camera: {}us.",
		master.get_color_image().get_device_timestamp().count(),
		sub.get_color_image().get_device_timestamp().count());
}

class MultiDevice {
public:
	MultiDevice(int32_t color_exposure_us, int32_t powerline_freq) {
		num_devices = k4a_device_get_installed_count();
		if (num_devices == 0) {
			throw new runtime_error("no device found.");
		}
		vector<int> device_indeces;
		for (int i = 0; i < num_devices; i++) {
			device_indeces.emplace_back(i);
		}
		bool master_found = false;
		if (device_indeces.size() == 0) {
			throw new runtime_error("There must be at least one camera.");
		}

		// open device(s)
		for (uint32_t i : device_indeces) {
			device d = device::open(i);

			// set the color exposure time and powerline freq (50Hz in UK)
			d.set_color_control(
				K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
				K4A_COLOR_CONTROL_MODE_MANUAL,
				color_exposure_us);
			d.set_color_control(
				K4A_COLOR_CONTROL_POWERLINE_FREQUENCY,
				K4A_COLOR_CONTROL_MODE_MANUAL,
				powerline_freq);	// 1 for 50Hz and 2 for 60Hz

			// find the Master device with sync_out_connected.
			// if there's only one device, just use it.
			if ((d.is_sync_out_connected() && !d.is_sync_in_connected()) || device_indeces.size() == 1) {
				master_device = move(d);
				master_found = true;
			}
			else if (
				(d.is_sync_out_connected() && d.is_sync_in_connected()) ||
				(!d.is_sync_out_connected() && d.is_sync_in_connected())
				) {
				sub_devices.emplace_back(move(d));
			}
			else {
				throw new runtime_error("wrong configuration or structure of camera(s)");
			}
		}
		if (!master_found) throw new runtime_error("No device with sync out connected found.");

		// build master and sub configs
		master_config = get_master_config();
		sub_configs = get_sub_configs();

		// start devices
		start_devices();

		// model calibrations for master and sub
		master_cal = master_device.get_calibration(master_config.depth_mode, master_config.color_resolution);
		for (int i = 0; i < sub_devices.size(); i++) {
			sub_cals.emplace_back(sub_devices[i].get_calibration(sub_configs[i].depth_mode, sub_configs[i].color_resolution));
		}
	}

	MultiDevice() : MultiDevice(8000, 1) {}

	// Need a calibration timeout (for calibrating 1 device).
	// The function will calibrate all sub devices.
	// `chessboard_pattern` in (h, w), default value cv::Size(9, 6).
	// `chessboard_square_len` in millimeter, default value 25.
	// `timeout` in second, default value 120.
	vector<TransformMatrix> calibrate(
		cv::Size chessboard_pattern = cv::Size(9, 6), 
		float chessboard_square_length = 25.0f, 
		double timeout = 120
	) {
		vector<TransformMatrix> tms;
		for (int i = 0; i < sub_devices.size(); i++) {
			TransformMatrix tm = calibrate_device(i, chessboard_pattern, chessboard_square_length, timeout);
			string file_name = fmt::format("Transform Matrix for sub device {}.json", i);
			tm.store_to_file(file_name);
			spdlog::info("Transform Matrix for sub device {} is generated and stored in {}", i, file_name);
			tms.emplace_back(tm);
		}
		return tms;
	}

	vector<capture> get_sync_captures(bool compare_sub_depth_instead_of_color = false) {
		vector<capture> captures(sub_devices.size() + 1);
		for (int i = 0; i < captures.size(); i++) {
			if (i == 0)
				master_device.get_capture(&captures[i]);
			else
				sub_devices[i - 1].get_capture(&captures[i]);
		}
		// just test. maybe delete this later.
		//return captures;

		// if no sub cameras, return captures which only has the master capture.
		if (sub_devices.empty())
			return captures;

		// check if getting the sync captures
		bool have_synced_images = false;
		chrono::system_clock::time_point start = chrono::system_clock::now();
		while (!have_synced_images) {
			int64_t duration_ms = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count();
			if (duration_ms > WAIT_FOR_SYNC_CAPTURE_TIMEOUT)
				throw new runtime_error("Timeout waiting for synchronized captures.");

			// get master color image time
			image master_color_image = captures[0].get_color_image();
			chrono::microseconds master_color_image_time = master_color_image.get_device_timestamp();

			// sub_device[i] <=> captures[i+1]
			for (int i = 1; i <= sub_devices.size(); i++) {
				image sub_image;
				if (compare_sub_depth_instead_of_color)
					sub_image = captures[i].get_depth_image();
				else
					sub_image = captures[i].get_color_image();

				if (master_color_image && sub_image) {
					chrono::microseconds sub_image_time = sub_image.get_device_timestamp();
					chrono::microseconds expected_sub_image_time =
						master_color_image_time +
						chrono::microseconds{ sub_configs[i - 1].subordinate_delay_off_master_usec } +
						chrono::microseconds{ sub_configs[i - 1].depth_delay_off_color_usec };
					chrono::microseconds sub_image_time_error = sub_image_time - expected_sub_image_time;

					if (sub_image_time_error < -MAX_ALLOWABLE_TIME_OFFSET_FOR_IMAGE_TIMESTAMP) {
						// sub image too old.
						log_lagging_time("sub", captures[0], captures[i]);
						sub_devices[i - 1].get_capture(&captures[i]);
						break;
					}
					else if (sub_image_time_error > MAX_ALLOWABLE_TIME_OFFSET_FOR_IMAGE_TIMESTAMP) {
						// master image too old.
						log_lagging_time("master", captures[0], captures[i]);
						master_device.get_capture(&captures[0]);
						break;
					}
					else {
						if (i == sub_devices.size()) {
							// all synchronized.
							log_sync_image_time(captures[0], captures[i]);
							have_synced_images = true;
						}
					}
				}
				else if (!master_color_image) {
					spdlog::info("no master image, recapturing...");
					master_device.get_capture(&captures[0]);
					break;
				}
				else if (!sub_image) {
					spdlog::info("no sub image, recapturing...");
					sub_devices[i - 1].get_capture(&captures[i]);
					break;
				}
			}
		}

		return captures;
	}

	/* getter */
	const device& get_master_device() const {
		return master_device;
	}

	const vector<device>& get_sub_devices() const {
		return sub_devices;
	}

	const device& get_sub_device_by_index(size_t i) const {
		if (i >= sub_devices.size()) {
			throw new runtime_error("Subordinate index too large.");
		}
		return sub_devices[i];
	}

	const calibration& get_master_calibration() const {
		return master_cal;
	}

	const vector<calibration>& get_sub_calibrations() const {
		return sub_cals;
	}

	const k4a_device_configuration_t& get_master_configuration() const {
		return master_config;
	}

	const vector<k4a_device_configuration_t>& get_sub_configurations() const {
		return sub_configs;
	}

	/* get transformation */
	transformation get_master_transformation() {
		return transformation(master_cal);
	}

	vector<transformation> get_sub_transformations() {
		vector<transformation> sub_transformations;
		for (calibration& cal : sub_cals) {
			sub_transformations.emplace_back(cal);
		}
		return sub_transformations;
	}

	vector<transformation> get_sub_color_to_master_depth_transformations(vector<TransformMatrix> tms) {
		vector<calibration> sub_color_to_master_depth_cals = generate_sub_color_to_master_depth_calibration(
			master_cal, sub_cals, tms
		);
		vector<transformation> transformations;
		for (calibration& c : sub_color_to_master_depth_cals) {
			transformations.emplace_back(c);
		}
		return transformations;
	}

	vector<transformation> get_sub_depth_to_master_color_transformations(vector<TransformMatrix> tms) {
		vector<calibration> sub_depth_to_master_color_cals = generate_sub_depth_to_master_color_calibration(
			master_cal,
			sub_cals,
			tms
		);
		vector<transformation> transformations;
		for (calibration& c : sub_depth_to_master_color_cals) {
			transformations.emplace_back(c);
		}
		return transformations;
	}


private:
	device master_device;
	vector<device> sub_devices;
	int num_devices;

	// configurations
	k4a_device_configuration_t master_config;
	vector<k4a_device_configuration_t> sub_configs;

	// calibrations for master and sub devices
	calibration master_cal;
	vector<calibration> sub_cals;

private:
	k4a_device_configuration_t get_default_config() {
		k4a_device_configuration_t c = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
		c.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
		c.color_resolution = K4A_COLOR_RESOLUTION_720P;
		c.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
		c.camera_fps = K4A_FRAMES_PER_SECOND_30;	// maybe limited usb bandwidth
		c.subordinate_delay_off_master_usec = 0;	// 0 for master
		c.synchronized_images_only = true;
		return c;
	}

	k4a_device_configuration_t get_master_config() {
		k4a_device_configuration_t c = get_default_config();
		if (num_devices == 1)
			c.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
		else
			c.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
		return c;
	}

	// corresponding sub_devices
	vector<k4a_device_configuration_t> get_sub_configs() {
		vector<k4a_device_configuration_t> cs;
		for (int i = 0; i < num_devices; i++) {
			k4a_device_configuration_t c = get_default_config();
			c.subordinate_delay_off_master_usec = MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_US * (i + 1);
			c.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
			cs.emplace_back(c);
		}
		return cs;
	}

	void start_devices() {
		// start sub devices first, then master.
		for (int i = 0; i < sub_devices.size(); i++) {
			sub_devices[i].start_cameras(&sub_configs[i]);
		}
		master_device.start_cameras(&master_config);
	}

private:
	// for calibration.

	cv::Matx33f calibration_to_color_camera_matrix(const calibration& c) {
		const k4a_calibration_intrinsic_parameters_t &params = c.color_camera_calibration.intrinsics.parameters;
		cv::Matx33f camera_matrix = cv::Matx33f::eye();
		camera_matrix(0, 0) = params.param.fx;
		camera_matrix(1, 1) = params.param.fy;
		camera_matrix(0, 2) = params.param.cx;
		camera_matrix(1, 2) = params.param.cy;
		return camera_matrix;
	}

	vector<float> calibration_to_color_camera_dist_coeffs(const calibration& c) {
		const k4a_calibration_intrinsic_parameters_t &params = c.color_camera_calibration.intrinsics.parameters;
		return { params.param.k1, params.param.k2, params.param.p1, params.param.p2 , params.param.k3 , params.param.k4 , params.param.k5 , params.param.k6 };
	}

	// call find_master_chessboard_corners first.
	bool find_chessboard_corners(
		const cv::Mat& master_color_image,
		const cv::Mat& sub_color_image,
		const cv::Size& chessboard_pattern,
		vector<cv::Point2f>& master_chessboard_corners,
		vector<cv::Point2f>& sub_chessboard_corners
	) {
		bool found_master_chessboard_corners = findChessboardCorners(
			master_color_image,
			chessboard_pattern,
			master_chessboard_corners
		);
		bool found_sub_chessborad_corners = findChessboardCorners(
			sub_color_image,
			chessboard_pattern,
			sub_chessboard_corners
		);
		if (!found_master_chessboard_corners) {
			spdlog::error("Cannot found the chessboard corners in master image.");
			return false;
		}
		if (!found_sub_chessborad_corners) {
			spdlog::error("Cannot found the chessboard corners in sub image.");
			return false;
		}

		// make sure the chessboard corners for master and sub have the same directions.
		// before applying this algorithm, the two cameras (master and sub) must be oriented in the same manner.
		cv::Vec2f master_image_corners = master_chessboard_corners.back() - master_chessboard_corners.front();
		cv::Vec2f sub_image_corners = sub_chessboard_corners.back() - sub_chessboard_corners.front();
		if (master_image_corners.dot(sub_image_corners) <= 0.0) {
			std::reverse(sub_chessboard_corners.begin(), sub_chessboard_corners.end());
		}
		return true;
	}

	// get the transform matrix (in 3d).
	TransformMatrix stereo_calibration(
		const calibration& master_cal,
		const calibration& sub_cal,
		const vector<vector<cv::Point2f>>& master_chessboard_corners_list,
		const vector<vector<cv::Point2f>>& sub_chessboard_corners_list,
		const cv::Size& image_size,
		const cv::Size& chessboard_pattern,
		float chessboard_square_len
	) {
		// let the top left corner be (0, 0)
		vector<cv::Point3f> chessboard_corners_coord;
		for (int h = 0; h < chessboard_pattern.height; ++h) {
			for (int w = 0; w < chessboard_pattern.width; ++w) {
				chessboard_corners_coord.emplace_back(
					cv::Point3f{ w * chessboard_square_len, h * chessboard_square_len, 0.0 }
				);
			}
		}
		vector<vector<cv::Point3f>> chessboard_corners_coords(master_chessboard_corners_list.size(), chessboard_corners_coord);

		cv::Matx33f master_camera_matrix = calibration_to_color_camera_matrix(master_cal);
		cv::Matx33f sub_camera_matrix = calibration_to_color_camera_matrix(sub_cal);
		vector<float> master_dist_coeffs = calibration_to_color_camera_dist_coeffs(master_cal);
		vector<float> sub_dist_coeffs = calibration_to_color_camera_dist_coeffs(sub_cal);

		TransformMatrix result;
		double error = stereoCalibrate(
			chessboard_corners_coords,
			sub_chessboard_corners_list,
			master_chessboard_corners_list,
			sub_camera_matrix,
			sub_dist_coeffs,
			master_camera_matrix,
			master_dist_coeffs,
			image_size,
			result.R,
			result.t,
			cv::noArray(),
			cv::noArray(),
			cv::CALIB_FIX_INTRINSIC | cv::CALIB_RATIONAL_MODEL | cv::CALIB_CB_FAST_CHECK
		);
		spdlog::info("Calibration Error: {}", error);
		TransformMatrix new_result = TransformMatrix(result.R, result.t);
		return new_result;
	}

	// calibration for one subordinate device.
	TransformMatrix calibrate_device(
		int sub_device_index,
		const cv::Size chessboard_pattern,
		float chessboard_square_len,
		double timeout
	) {
		calibration sub_cal = sub_cals[sub_device_index];

		vector<vector<cv::Point2f>> master_chessboard_corners_list;
		vector<vector<cv::Point2f>> sub_chessboard_corners_list;

		chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
		while (chrono::duration<double>(chrono::system_clock::now() - start_time).count() < timeout) {
			vector<capture> captures = get_sync_captures();
			capture master_capture = captures[0];
			capture sub_capture = captures[sub_device_index + 1];
			image master_color_image = master_capture.get_color_image();
			image sub_color_image = sub_capture.get_color_image();
			cv::Mat master_color_image_mat = color_to_opencv(master_color_image);
			cv::Mat sub_color_image_mat = color_to_opencv(sub_color_image);

			vector<cv::Point2f> master_chessboard_corners;
			vector<cv::Point2f> sub_chessboard_corners;
			bool get_corners = find_chessboard_corners(
				master_color_image_mat,
				sub_color_image_mat,
				chessboard_pattern,
				master_chessboard_corners,
				sub_chessboard_corners
			);
			//bool get_corners = false;
			if (get_corners) {
				master_chessboard_corners_list.emplace_back(master_chessboard_corners);
				sub_chessboard_corners_list.emplace_back(sub_chessboard_corners);
				drawChessboardCorners(master_color_image_mat, chessboard_pattern, master_chessboard_corners, true);
				drawChessboardCorners(sub_color_image_mat, chessboard_pattern, sub_chessboard_corners, true);
			}

			cv::imshow("Master Camera", master_color_image_mat);
			cv::waitKey(1);
			cv::imshow("Subordinate Camera", sub_color_image_mat);
			cv::waitKey(1);

			if (master_chessboard_corners_list.size() >= 30) {
				spdlog::info("Calculating calibration for subordinate device {}", sub_device_index);
				return stereo_calibration(
					master_cal,
					sub_cal,
					master_chessboard_corners_list,
					sub_chessboard_corners_list,
					master_color_image_mat.size(),
					chessboard_pattern,
					chessboard_square_len
				);
			}
		}
		spdlog::error("Calibration timeout while calibrating subordinate device {}", sub_device_index);
		throw new runtime_error("Calibration timeout");
	}

	// sub depth -> sub color -> master color
	vector<calibration> generate_sub_depth_to_master_color_calibration(
		const calibration& master_cal,
		vector<calibration> sub_cals,
		vector<TransformMatrix> sub_color_to_master_color_tms
	) {
		vector<calibration> cals;
		for (int i = 0; i < sub_color_to_master_color_tms.size(); i++) {
			// sub depth -> sub color
			const k4a_calibration_extrinsics_t& ex = sub_cals[i].extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
			TransformMatrix sub_depth_to_sub_color_tm;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					sub_depth_to_sub_color_tm.R(i, j) = ex.rotation[i * 3 + j];
				}
			}
			sub_depth_to_sub_color_tm.t = cv::Vec3d(ex.translation[0], ex.translation[1], ex.translation[2]);

			// sub color -> master color
			TransformMatrix sub_color_to_master_color_tm = sub_color_to_master_color_tms[i];

			// compose
			TransformMatrix sub_depth_to_master_color_tm = sub_depth_to_sub_color_tm.compose(sub_color_to_master_color_tm);

			// generate calibration
			calibration c = sub_cals[i];
			k4a_calibration_extrinsics_t& ex2 = c.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					ex2.rotation[i * 3 + j] = static_cast<float>(sub_depth_to_master_color_tm.R(i, j));
				}
			}
			for (int i = 0; i < 3; i++) {
				ex2.translation[i] = static_cast<float>(sub_depth_to_master_color_tm.t[i]);
			}
			c.color_camera_calibration = master_cal.color_camera_calibration;
			cals.emplace_back(c);
		}
		return cals;
	}

	// sub color -> master color -> master depth
	vector<calibration> generate_sub_color_to_master_depth_calibration(
		calibration& master_cal,
		vector<calibration> sub_cals,
		vector<TransformMatrix> sub_color_to_master_color_tms
	) {
		vector<calibration> cals;
		for (int i = 0; i < sub_cals.size(); i++) {
			// sub color (2d) -> sub color (3d)
			// don't have to modify sub_cals[i]

			// sub color (3d) -> master color (3d) -> master depth (3d)
			TransformMatrix sub_color_to_master_color_tm = sub_color_to_master_color_tms[i];

			const k4a_calibration_extrinsics_t& ex = sub_cals[i].extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH];
			TransformMatrix master_color_to_master_depth_tm;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					master_color_to_master_depth_tm.R(i, j) = ex.rotation[i * 3 + j];
				}
			}
			master_color_to_master_depth_tm.t = cv::Vec3d(ex.translation[0], ex.translation[1], ex.translation[2]);

			TransformMatrix sub_color_to_master_depth_tm = sub_color_to_master_color_tm.compose(master_color_to_master_depth_tm);

			// generate calibration
			calibration c = sub_cals[i];
			k4a_calibration_extrinsics_t& ex2 = c.extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH];
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					ex2.rotation[i * 3 + j] = static_cast<float>(sub_color_to_master_depth_tm.R(i, j));
				}
			}
			for (int i = 0; i < 3; i++) {
				ex2.translation[i] = static_cast<float>(sub_color_to_master_depth_tm.t[i]);
			}
			c.depth_camera_calibration = master_cal.depth_camera_calibration;
			cals.emplace_back(c);
		}
		return cals;
	}
};

