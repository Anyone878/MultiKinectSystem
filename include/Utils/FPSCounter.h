#pragma once

#include <ctime>
#include <chrono>

typedef std::chrono::steady_clock::time_point time_point;

class FPSCounter {
public:
	FPSCounter() : startTime(std::chrono::high_resolution_clock::now()), numFrames(0), fps(0) {}
	void start();
	uint32_t endAndGetFPS();

private:
	time_point startTime;
	int numFrames;
	uint32_t fps;
};

void FPSCounter::start()
{
	if (numFrames == 0) startTime = std::chrono::high_resolution_clock::now();
}

uint32_t FPSCounter::endAndGetFPS()
{
	++numFrames;
	if (numFrames == 10) {
		time_point endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
		fps = static_cast<uint32_t>(1000000 / (duration / numFrames));
		numFrames = 0;
	}
	return fps;
}