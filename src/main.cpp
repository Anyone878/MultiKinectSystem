#include <string>
#include <chrono>
#include <spdlog/spdlog.h>
#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>
#include <csignal>

#include <draco/point_cloud/point_cloud_builder.h>
#include <draco/point_cloud/point_cloud.h>
#include <draco/compression/encode.h>
#include <draco/core/encoder_buffer.h>
#include <draco/core/status.h>
#include <draco/metadata/geometry_metadata.h>

#include <shared_mutex>
#include <BS_thread_pool.hpp>
#include <BS_thread_pool_utils.hpp>

#include <enet/enet.h>

#include "../include/Utils/FPSCounter.h"
#include "../include/Utils/MatrixUtils.h"
#include "../include/Utils/ColorChart.h"
#include "../include/Utils/Timer.h"
#include "../include/Utils/ThreadSafeQueue.h"

#include "../include/Cuda/CublasHandleManager.cuh"
#include "../include/Cuda/CudaStreamManager.cuh"
#include "../include/Cuda/CudaTransform.cuh"
#include "../include/Cuda/CudaUtils.cuh"
#include "../include/Cuda/CudaVoxelGridFilter.cuh"

#include "../include/OpenGLFramework.h"
#include "../include/MultiDevice.h"
#include "../include/MultiDeviceTracker.h"
#include "../include/PointCloud.h"

constexpr auto QUEUE_MAX_SIZE = 30;

using namespace std;
using namespace cv;
using namespace k4a;

static void convert_3d_to_3d_for_body(k4abt_body_t& body, calibration cal, k4a_calibration_type_t src_type, k4a_calibration_type_t dist_type);
static void body_transformation(k4abt_body_t& body, TransformMatrix transformation_matrix);
static k4abt_body_t merge_bodies(vector<k4abt_body_t> tidy_bodies);

std::atomic<bool> should_stop(false);

int main() {

	FPSCounter fpsCounter;

	MultiDevice multi_device = MultiDevice();

	k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
	tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_TENSORRT;
	MultiDeviceTracker multi_device_tracker = MultiDeviceTracker(multi_device, tracker_config);

	vector<TransformMatrix> sub_color_to_master_color_tms = multi_device.calibrate(cv::Size(5, 3), 95.f);
	/*vector<TransformMatrix> sub_color_to_master_color_tms;
	sub_color_to_master_color_tms.emplace_back("E:\\Projects\\kinect\\KinectApp2019\\Transform Matrix for sub device 0.json");*/
	// init transform matrix 
	// for sub-to-master: sub-depth -> sub-color -> master-color -> master-depth
	calibration master_cal = multi_device.get_master_calibration();
	vector<calibration> sub_cals = multi_device.get_sub_calibrations();
	TransformMatrix mc2md(master_cal, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH);
	vector<TransformMatrix> subd2subc;
	for (calibration sub_cal : sub_cals) {
		subd2subc.emplace_back(sub_cal, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR);
	}
	// calculate the final transform matrix
	vector<TransformMatrix> sd2md_tms;
	sd2md_tms.emplace_back();
	for (int i = 0; i < subd2subc.size(); ++i) {
		sd2md_tms.push_back(mc2md.compose(sub_color_to_master_color_tms[i]).compose(subd2subc[i]));
	}

	// cuda streams
	CudaStreamManager& stream_manager = CudaStreamManager::get_instance();
	stream_manager.initialize_streams(sub_cals.size() + 1);

	// cublas handle
	CublasHandleManager::get_instance().initialize_handles(sub_cals.size() + 1);

	// point cloud compression (draco)
	draco::PointCloudBuilder pc_builder;
	constexpr int compression_method = draco::POINT_CLOUD_SEQUENTIAL_ENCODING;
	draco::Encoder pc_encoder; 
	pc_encoder.SetEncodingMethod(compression_method);
	pc_encoder.SetSpeedOptions(10, 10);

	// get transformations
	transformation master_transformation = multi_device.get_master_transformation();
	vector<transformation> sub_transformations = multi_device.get_sub_transformations();
	vector<transformation> transformations;
	for (int j = 0; j < sub_transformations.size() + 1; ++j) {
		if (j == 0)
			transformations.emplace_back(master_cal);
		else
			transformations.emplace_back(sub_cals[j - 1]);
	}

	namedWindow("Test", WINDOW_NORMAL);
	OpenGLFramework app;
	if (!app.init()) {
		spdlog::error("App init failed.");
		return -1;
	}
	vector<OpenGLFramework::PointSet> point_sets;
	vector<OpenGLFramework::JointSet> joint_sets;
	point_sets.emplace_back();

	// init thread pool
	BS::thread_pool pool(std::thread::hardware_concurrency());
	shared_mutex mtx_sd2md_tms;
	BS::synced_stream sync_out;
	// all results are sync by timestamp
	My::ThreadSafeQueue<std::pair<std::vector<k4abt::frame>, chrono::steady_clock::time_point>> cap2bodytracking_queue(QUEUE_MAX_SIZE);
	My::ThreadSafeQueue<std::pair<std::vector<k4abt_body_t>, chrono::steady_clock::time_point>> body_tracking_output_queue(QUEUE_MAX_SIZE);
	My::ThreadSafeQueue<std::pair<std::vector<capture>, chrono::steady_clock::time_point>> cap2transformation_queue(QUEUE_MAX_SIZE);
	My::ThreadSafeQueue<std::pair<std::vector<My::PointCloud>, chrono::steady_clock::time_point>> trans2data_reduction_queue(QUEUE_MAX_SIZE);
	My::ThreadSafeQueue<std::pair<std::vector<My::PointColorRGB>, chrono::steady_clock::time_point>> reduction2sync_queue(QUEUE_MAX_SIZE);
	My::ThreadSafeQueue<std::pair<std::vector<My::PointColorRGB>, std::vector<k4abt_body_t>>> sync_output_queue(QUEUE_MAX_SIZE);
	My::ThreadSafeQueue<draco::EncoderBuffer> compression2transmission_queue(3);

	std::signal(
		SIGINT,
		[](int signum) {
			spdlog::info("Interrupt Signal ({}) received.", signum);
			spdlog::info("Stoping...");
			should_stop = true;
		}
	);

	/* Capture thread No.1 */
	pool.detach_task(
		[
			&cap2bodytracking_queue,
			&cap2transformation_queue,
			&multi_device,
			&multi_device_tracker,
			&sync_out
		]() {
			My::Timer timer;
			int iteration_count = 0;
			while (!should_stop) {
				++iteration_count;
				timer.start();

				vector<capture> capture_0 = multi_device.get_sync_captures();
				multi_device_tracker.enqueue_sync_captures(capture_0);
				for (capture& c : capture_0)
					c.reset();

				vector<frame> frames = multi_device_tracker.pop_frames();
				chrono::steady_clock::time_point ts = chrono::steady_clock::now();
				pair<vector<capture>, chrono::steady_clock::time_point> captures_with_timestamp{
					multi_device_tracker.get_original_captures_in_frames(frames),
					ts
				};
				cap2transformation_queue.push(std::move(captures_with_timestamp));
				pair<vector<frame>, chrono::steady_clock::time_point> frames_with_timestamp{
					std::move(frames),
					ts
				};
				cap2bodytracking_queue.push(std::move(frames_with_timestamp)); // get body tracking frames

				timer.pause();
				if (iteration_count >= 20) {
					spdlog::info("[CAPTURE] {} FPS.", 1000 / (timer.elapsed() / iteration_count));
					timer.reset();
					iteration_count = 0;
				}
			}
		}
	);

	/* Body Tracking Thread No.2.1 */
	pool.detach_task(
		[
			&cap2bodytracking_queue,
			&body_tracking_output_queue,
			&sd2md_tms,
			&mtx_sd2md_tms,
			&sync_out
		]() {
			My::Timer timer;
			int iteration_count = 0;
			while (!should_stop) {
				++iteration_count;
				timer.start();

				pair<vector<frame>, chrono::steady_clock::time_point> frames_with_timestamp = cap2bodytracking_queue.pop();
				vector<frame>& frames = frames_with_timestamp.first;
				chrono::steady_clock::time_point ts = frames_with_timestamp.second;
				vector<vector<k4abt_body_t>> tidy;
				for (int i = 0; i < frames.size(); ++i) {
					// for each stream
					for (uint32_t j = 0; j < frames[i].get_num_bodies(); ++j) {
						// each body in the stream
						k4abt_body_t b = frames[i].get_body(j);
						if (i == 0) {
							vector<k4abt_body_t> bs;
							bs.emplace_back(b);
							tidy.emplace_back(bs);
						}
						else {
							bool added = false;

							mtx_sd2md_tms.lock_shared();
							body_transformation(b, sd2md_tms[i]);
							mtx_sd2md_tms.unlock_shared();

							for (vector<k4abt_body_t>& row : tidy) {
								if (MultiDeviceTracker::compare_bodies(row[0], b, 100.f, 5)) {
									row.emplace_back(b);
									added = true;
								}
							}
							if (!added) {
								vector<k4abt_body_t> bs;
								bs.emplace_back(b);
								tidy.emplace_back(bs);
							}
						}
					}
				}
				vector<k4abt_body_t> main_bodies;
				for (vector<k4abt_body_t>& v : tidy) {
					main_bodies.emplace_back(merge_bodies(v));
				}
				body_tracking_output_queue.push(std::move(std::make_pair(std::move(main_bodies), ts)));

				timer.pause();
				if (iteration_count >= 20) {
					spdlog::info("[BODY TRACKING] {} FPS.", 1000 / (timer.elapsed() / iteration_count));
					iteration_count = 0;
					timer.reset();
				}
			}
		}
	);

	/* Point Cloud Transformation Thread No.2.2 */
	pool.detach_task(
		[
			&cap2transformation_queue,
			&trans2data_reduction_queue,
			&pool,
			&transformations,
			&sd2md_tms,
			&mtx_sd2md_tms,
			&sync_out
		]() {
			My::Timer timer;
			int iteration_count = 0;

			while (!should_stop) {
				++iteration_count;
				timer.start();

				pair<vector<capture>, chrono::steady_clock::time_point> captures_with_timestamp = cap2transformation_queue.pop();
				vector<capture>& captures = captures_with_timestamp.first;
				chrono::steady_clock::time_point ts = captures_with_timestamp.second;

				vector<My::PointCloud> point_clouds(captures.size());
				for (int j = 0; j < captures.size(); ++j) {
					image color_image = captures[j].get_color_image();
					image depth_image = captures[j].get_depth_image();
					int image_size = depth_image.get_width_pixels() * depth_image.get_height_pixels();
					image color_to_depth_image = transformations[j].color_image_to_depth_camera(depth_image, color_image);
					image depth_to_point_cloud = transformations[j].depth_image_to_point_cloud(depth_image, K4A_CALIBRATION_TYPE_DEPTH);
					point_clouds[j].points.resize(image_size);
					std::copy(
						reinterpret_cast<My::Point*>(depth_to_point_cloud.get_buffer()),
						reinterpret_cast<My::Point*>(depth_to_point_cloud.get_buffer()) + image_size,
						point_clouds[j].points.begin()
					);
					point_clouds[j].colors.resize(image_size);
					std::copy(
						reinterpret_cast<My::ColorBGRA*>(color_to_depth_image.get_buffer()),
						reinterpret_cast<My::ColorBGRA*>(color_to_depth_image.get_buffer()) + image_size,
						point_clouds[j].colors.begin()
					);
				}

				//vector<My::PointCloud> outputs;
				//for (int i = 0; i < captures.size(); ++i) {
				//	outputs.push_back(
				//		cuda_transformation(
				//			point_clouds[i],
				//			sd2md_tms[i].to_homogeneous(),
				//			0
				//		)
				//	);
				//}

				BS::multi_future<My::PointCloud> futures = pool.submit_sequence<int>(0, captures.size(),
					[
						&point_clouds,
						&sd2md_tms,
						&mtx_sd2md_tms
					](const int index) {
						mtx_sd2md_tms.lock_shared();
						return cuda_transformation(
							point_clouds[index],
							sd2md_tms[index].to_homogeneous(),
							index
						);
					}
				);
				trans2data_reduction_queue.push(std::move(
					std::make_pair(
						std::move(futures.get()),
						ts
					)
				));

				timer.pause();
				if (iteration_count >= 20) {
					spdlog::info("[TRANSFORMATION] {} FPS.", 1000 / (timer.elapsed() / iteration_count));
					timer.reset();
					iteration_count = 0;
				}
			}
		}
	);

	/* Data Reduction Thread No.3 */
	pool.detach_task(
		[
			&trans2data_reduction_queue,
			&reduction2sync_queue,
			&sync_out
		]() {
			My::Timer timer;
			int iteration_count = 0;
			while (!should_stop) {
				++iteration_count;
				timer.start();

				pair<vector<My::PointCloud>, chrono::steady_clock::time_point> transformed_point_clouds_with_timestamp = trans2data_reduction_queue.pop();
				vector<My::PointCloud>& transformed_point_clouds = transformed_point_clouds_with_timestamp.first;
				chrono::steady_clock::time_point ts = transformed_point_clouds_with_timestamp.second;

				// merge points and colors into one vector
				int original_image_size = transformed_point_clouds[0].points.size();
				int new_image_size = original_image_size * transformed_point_clouds.size();
				// merge color
				vector<My::ColorBGRA> complete_color;
				complete_color.reserve(new_image_size);
				for (int j = 0; j < transformed_point_clouds.size(); ++j) {
					copy(
						transformed_point_clouds[j].colors.begin(),
						transformed_point_clouds[j].colors.end(),
						back_inserter(complete_color)
					);
				}
				// merge point
				My::PointCloud& complete_point = transformed_point_clouds[0];
				complete_point.points.reserve(new_image_size);
				for (int j = 1; j < transformed_point_clouds.size(); ++j) {
					copy(
						transformed_point_clouds[j].points.begin(),
						transformed_point_clouds[j].points.end(),
						back_inserter(complete_point.points)
					);
				}
				vector<My::PointColorRGB> output;
				cuda_voxel_grid_filter(
					complete_point,
					complete_color,
					5.f,
					output
				);
				reduction2sync_queue.push(std::move(
					std::make_pair(
						std::move(output),
						ts
					)
				));

				timer.pause();
				if (iteration_count >= 20) {
					spdlog::info("[DATA REDUCTION] {} FPS", 1000 / (timer.elapsed() / iteration_count));
					iteration_count = 0;
					timer.reset();
				}
			}
		}
	);

	/* Data Sync Thread No.4 */
	pool.detach_task(
		[
			&body_tracking_output_queue,
			&reduction2sync_queue,
			&sync_output_queue
		]() {
			My::Timer timer;
			int iteration_count = 0;

			while (!should_stop) {
				++iteration_count;
				timer.start();
				std::pair<std::vector<My::PointColorRGB>, chrono::steady_clock::time_point> output_with_timestamp = reduction2sync_queue.pop();
				std::pair<std::vector<k4abt_body_t>, chrono::steady_clock::time_point> bodytracking_output_with_timestamp = body_tracking_output_queue.pop();
				while (output_with_timestamp.second != bodytracking_output_with_timestamp.second) {
					if (output_with_timestamp.second > bodytracking_output_with_timestamp.second) {
						// wait for an updated body tracking result
						bodytracking_output_with_timestamp = body_tracking_output_queue.pop();
					}
					else {
						output_with_timestamp = reduction2sync_queue.pop();
					}
				}
				sync_output_queue.push(std::move(
					std::make_pair(
						std::move(output_with_timestamp.first),
						std::move(bodytracking_output_with_timestamp.first)
					)
				));

				if (iteration_count >= 20) {
					spdlog::info("[DATA SYNC] {} FPS.", 1000 / (timer.elapsed() / iteration_count));
					iteration_count = 0;
					timer.reset();
				}
			}
		}
	);

	/* Data Compression Thread No.5 */
	pool.detach_task(
		[
			&sync_output_queue,
			&compression2transmission_queue,
			&pc_builder,
			&pc_encoder
		]() {
			My::Timer timer;
			int iteration_count = 0;

			while (!should_stop) {
				++iteration_count;
				timer.start();

				pair<vector<My::PointColorRGB>, vector<k4abt_body_t>> output_0 = sync_output_queue.pop();
				vector<My::PointColorRGB> output = output_0.first;
				vector<k4abt_body_t> body_tracking_output = output_0.second;
				vector<My::Point> new_output_point;
				vector<My::ColorRGB> new_output_colorRGB;
				cuda_unzip_point_color(output, new_output_point, new_output_colorRGB);

				pc_builder.Start(new_output_point.size());
				const int point_attr_id = pc_builder.AddAttribute(
					draco::GeometryAttribute::POSITION,
					3,
					draco::DT_INT16,
					false
				);
				const int color_attr_id = pc_builder.AddAttribute(
					draco::GeometryAttribute::COLOR,
					3,
					draco::DT_UINT8,
					false
				);
				pc_builder.SetAttributeValuesForAllPoints(
					point_attr_id,
					new_output_point.data(),
					3 * sizeof(int16_t)
				);
				pc_builder.SetAttributeValuesForAllPoints(
					color_attr_id,
					new_output_colorRGB.data(),
					3 * sizeof(uint8_t)
				);
				unique_ptr<draco::PointCloud> pc_geometry = pc_builder.Finalize(false);
				draco::GeometryMetadata metadata;
				for (int i = 0; i < body_tracking_output.size(); ++i) {
					vector<uint8_t> raw_body_tracking_data(sizeof(k4abt_body_t));
					std::copy(
						reinterpret_cast<uint8_t*>(&body_tracking_output[i]),
						reinterpret_cast<uint8_t*>(&body_tracking_output[i]) + sizeof(k4abt_body_t),
						raw_body_tracking_data.begin()
					);
					metadata.AddEntryBinary(
						"Body Tracking Result " + std::to_string(i),
						raw_body_tracking_data
					);
				}
				pc_geometry->AddMetadata(std::make_unique<draco::GeometryMetadata>(metadata));
				draco::EncoderBuffer buffer;
				draco::Status status = pc_encoder.EncodePointCloudToBuffer(*pc_geometry, &buffer);
				if (status.ok()) {
					spdlog::info("size of the buffer: {} KB.", buffer.size() / 1024.f);
					compression2transmission_queue.push(std::move(buffer));
				}
				else {
					spdlog::info("compression error: {}.");
				}
				timer.pause();

				if (iteration_count >= 20) {
					spdlog::info("[DATA COMPRESSION] {} FPS.", 1000 / (timer.elapsed() / iteration_count));
					iteration_count = 0;
					timer.reset();
				}
			}
		}
	);

	// transformission init
	if (enet_initialize() != 0) {
		cout << "Init error." << endl;
		return EXIT_FAILURE;
	}
	atexit(enet_deinitialize); 

	ENetAddress address;
	ENetHost* p_server;
	std::mutex mtx_p_server; 
	//enet_address_set_host_ip(&address, "192.168.0.233");
	address.host = ENET_HOST_ANY;
	address.port = 7345;
	p_server = enet_host_create(
		&address,
		10,
		5,
		0, 0
	);
	if (p_server == NULL) {
		cout << "Create server error" << endl;
		exit(EXIT_FAILURE);
	}

	ENetEvent event;
	vector<ENetPeer*> p_peers;

	pool.detach_task(
		[
			&compression2transmission_queue,
			&p_server,
			&mtx_p_server
		]() {
			int iteration_count = 0;
			My::Timer timer;

			while (!should_stop) {
				++iteration_count;
				timer.start();

				draco::EncoderBuffer buffer = compression2transmission_queue.pop();
				const char* raw_data = buffer.data();
				ENetPacket* p = enet_packet_create(raw_data, buffer.size(), ENET_PACKET_FLAG_UNRELIABLE_FRAGMENT);

				mtx_p_server.lock();
				enet_host_broadcast(p_server, iteration_count % 5, p);
				mtx_p_server.unlock();

				timer.pause();
				if (iteration_count >= 5) {
					spdlog::info("[NETWORK TRANSMISSION] {} FPS.", 1000 / (timer.elapsed() / iteration_count));
					iteration_count = 0;
					timer.reset();
				}
			}
		}
	);

	while (!should_stop) {
		// wait events
		mtx_p_server.lock();
		int j = enet_host_service(p_server, &event, 0);
		mtx_p_server.unlock();
		if (j > 0) {
			switch (event.type) {
			case ENET_EVENT_TYPE_CONNECT:
				spdlog::info("CONNECT: {}.{}.{}.{}:{}",
					(event.peer->address.host & 0xFF),
					((event.peer->address.host >> 8) & 0xFF),
					((event.peer->address.host >> 16) & 0xFF),
					((event.peer->address.host >> 24) & 0xFF),
					event.peer->address.port);
				p_peers.push_back(event.peer);
				break;
			case ENET_EVENT_TYPE_DISCONNECT:
				spdlog::info("DISCONNECT: {}.{}.{}.{}:{}",
					(event.peer->address.host & 0xFF),
					((event.peer->address.host >> 8) & 0xFF),
					((event.peer->address.host >> 16) & 0xFF),
					((event.peer->address.host >> 24) & 0xFF),
					event.peer->address.port);
				p_peers.erase(remove(p_peers.begin(), p_peers.end(), event.peer), p_peers.end());
				break;
			}
		}
		this_thread::sleep_for(10ms);
	}

	// capture and display
	//while (!app.windowShouldClose()) {
	//	fpsCounter.start();

		//// display
		//for (int i = 0; i < body_tracking_output.size(); ++i) {
		//	joint_sets.emplace_back();
		//	for (int j = 0; j < K4ABT_JOINT_COUNT; ++j) {
		//		joint_sets[i].add_joint(
		//			glm::vec3(
		//				body_tracking_output[i].skeleton.joints[j].position.xyz.x,
		//				body_tracking_output[i].skeleton.joints[j].position.xyz.y,
		//				body_tracking_output[i].skeleton.joints[j].position.xyz.z
		//			),
		//			COLOR_CHART[i % COLOR_CHART_COUNT]
		//		);
		//	}
		//}
		//point_sets[0].update(
		//	reinterpret_cast<int16_t*>(new_output_point.data()),
		//	reinterpret_cast<uint8_t*>(new_output_colorBGRA.data()),
		//	new_output_point.size()
		//);

		//app.beforeUpdate();
		//app.updatePointCloud(point_sets);
		//app.updateJoints(joint_sets);
		//app.afterUpdate();

		//for (OpenGLFramework::JointSet& js : joint_sets) {
		//	js.release_joints();
		//}

		//device->release_images();
	//	spdlog::info("[DISPLAY] FPS: {}", fpsCounter.endAndGetFPS());
	//}
	//should_stop = true;

	return 0;
}

static void convert_3d_to_3d_for_body(k4abt_body_t& body, calibration cal, k4a_calibration_type_t src_type, k4a_calibration_type_t dst_type)
{
	k4abt_joint_t* j = body.skeleton.joints;
	for (int i = 0; i < K4ABT_JOINT_COUNT; ++i) {
		j[i].position = cal.convert_3d_to_3d(j[i].position, src_type, dst_type);
	}
}

void body_transformation(k4abt_body_t& body, TransformMatrix transformation_matrix)
{
	k4abt_joint_t* joints = body.skeleton.joints;
	for (int i = 0; i < K4ABT_JOINT_COUNT; ++i) {
		cv::Vec4d cvt_pos = transformation_matrix.dot(cv::Vec4d(joints[i].position.xyz.x, joints[i].position.xyz.y, joints[i].position.xyz.z, 1.0f));
		k4a_float3_t new_pos{
			static_cast<float>(cvt_pos[0]),
			static_cast<float>(cvt_pos[1]),
			static_cast<float>(cvt_pos[2])
		};
		joints[i].position = new_pos;
	}
}

// average
k4abt_body_t merge_bodies(vector<k4abt_body_t> tidy_bodies)
{
	k4abt_body_t new_body = tidy_bodies[0];
	vector<int> divisors(K4ABT_JOINT_COUNT, 1);
	for (int i = 1; i < tidy_bodies.size(); ++i) {
		for (int j = 0; j < K4ABT_JOINT_COUNT; ++j) {
			if (tidy_bodies[i].skeleton.joints[j].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW) {
				if (new_body.skeleton.joints[j].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW) {
					++divisors[j];
					new_body.skeleton.joints[j].position.v[0] += tidy_bodies[i].skeleton.joints[j].position.v[0];
					new_body.skeleton.joints[j].position.v[1] += tidy_bodies[i].skeleton.joints[j].position.v[1];
					new_body.skeleton.joints[j].position.v[2] += tidy_bodies[i].skeleton.joints[j].position.v[2];
				}
				else {
					new_body.skeleton.joints[j].confidence_level = tidy_bodies[i].skeleton.joints[j].confidence_level;
					new_body.skeleton.joints[j].position.v[0] = tidy_bodies[i].skeleton.joints[j].position.v[0];
					new_body.skeleton.joints[j].position.v[1] = tidy_bodies[i].skeleton.joints[j].position.v[1];
					new_body.skeleton.joints[j].position.v[2] = tidy_bodies[i].skeleton.joints[j].position.v[2];
				}
			}
		}
	}
	for (int i = 0; i < K4ABT_JOINT_COUNT; ++i) {
		new_body.skeleton.joints[i].position.v[0] /= static_cast<float>(divisors[i]);
		new_body.skeleton.joints[i].position.v[1] /= static_cast<float>(divisors[i]);
		new_body.skeleton.joints[i].position.v[2] /= static_cast<float>(divisors[i]);
	}
	return new_body;
}
