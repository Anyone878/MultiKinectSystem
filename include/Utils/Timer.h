#pragma once

#include <chrono>

namespace My {
	class Timer {
	public:
		Timer() : is_running(false), elapsed_time(0) {}

		void start() {
			if (!is_running) {
				is_running = true;
				start_time = std::chrono::steady_clock::now();
			}
		}

		void pause() {
			if (is_running) {
				auto now = std::chrono::steady_clock::now();
				elapsed_time += std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
				is_running = false;
			}
		}

		// return in ms
		long long elapsed() {
			if (is_running) {
				auto now = std::chrono::steady_clock::now();
				return elapsed_time + std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
			}
			else {
				return elapsed_time;
			}
		}


		void reset() {
			is_running = false;
			elapsed_time = 0;
		}

	private:
		bool is_running;
		std::chrono::steady_clock::time_point start_time;
		long long elapsed_time;
	};
}