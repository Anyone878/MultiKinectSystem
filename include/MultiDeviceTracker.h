#pragma once

#include <vector>
#include <chrono>

#include "MultiDevice.h"
#include "./Utils/MatrixUtils.h"
#include <k4abt.hpp>

using namespace std;
using namespace k4abt;


// only support one camera
// TODO: support for multiple cameras
class MultiDeviceTracker {
public:	// constructor
	MultiDeviceTracker(const MultiDevice& multi_device, k4abt_tracker_configuration_t tracker_configuration = K4ABT_TRACKER_CONFIG_DEFAULT, bool test = false, int num_devices = 0) {
		const device& master_device = multi_device.get_master_device();
		trackers.emplace_back(tracker::create(multi_device.get_master_calibration(), tracker_configuration));
		if (!test) {
			const vector<calibration>& sub_cals = multi_device.get_sub_calibrations();
			for (const calibration& sub_cal : sub_cals) {
				trackers.emplace_back(tracker::create(sub_cal, tracker_configuration));
			}
		}
		else {
			for (int i = 0; i < num_devices; ++i) {
				trackers.emplace_back(tracker::create(multi_device.get_master_calibration(), tracker_configuration));
			}
		}
	}

public:	// processing trackers
	bool enqueue_sync_captures(vector<capture> captures, chrono::milliseconds timeout = chrono::milliseconds(K4A_WAIT_INFINITE)) {
		if (trackers.size() != captures.size()) throw new runtime_error("trackers.size() != captures.size()");
		for (int i = 0; i < trackers.size(); i++) {
			if (!trackers[i].enqueue_capture(captures[i], timeout)) return false;
		}
		return true;
	}

	vector<frame> pop_frames(chrono::milliseconds timeout = chrono::milliseconds(K4A_WAIT_INFINITE)) const {
		vector<frame> frames;
		for (const tracker& t : trackers) {
			frames.emplace_back(t.pop_result(timeout));
		}
		return frames;
	}

public:
	static vector<vector<k4abt_body_t>> get_bodies_in_frames(const vector<frame>& frames) {
		vector<vector<k4abt_body_t>> bodies;
		for (const frame& f : frames) {
			int num_bodies = f.get_num_bodies();
			vector<k4abt_body_t> b;
			for (int i = 0; i < num_bodies; ++i) {
				b.emplace_back(f.get_body(i));
			}
			bodies.emplace_back(b);
		}
		return bodies;
	}

	// `distance_threshold` in mm
	static bool compare_bodies(const k4abt_body_t b1, const k4abt_body_t b2, float distance_threshold, int count_threshold) {
		return compare_bodies(b1.skeleton, b2.skeleton, distance_threshold, count_threshold);
	}

	// `distance_threshold` in mm
	static bool compare_bodies(const k4abt_skeleton_t s1, const k4abt_skeleton_t s2, float distance_threshold, int count_threshold) {
		int count = 0;
		for (int i = 0; i < K4ABT_JOINT_COUNT; ++i) {
			if (distance(
				glm::vec3(s1.joints[i].position.v[0], s1.joints[i].position.v[1], s1.joints[i].position.v[2]),
				glm::vec3(s2.joints[i].position.v[0], s2.joints[i].position.v[1], s2.joints[i].position.v[2])
			) <= distance_threshold) {
				++count;
			}
		}
		if (count >= count_threshold)
			return true;
		else
			return false;
	}

	static vector<capture> get_original_captures_in_frames(const vector<frame>& frames) {
		vector<capture> captures;
		for (const frame& f : frames) {
			captures.emplace_back(f.get_capture());
		}
		return captures;
	}

private:	// properties
	vector<tracker> trackers;
};