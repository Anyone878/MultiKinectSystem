#pragma once

#include <glm/glm.hpp>

constexpr size_t COLOR_CHART_COUNT = 13;
constexpr glm::vec4 COLOR_CHART[COLOR_CHART_COUNT] = {
	glm::vec4(255.f, 102.f, 102.f, 255.f) / 255.f,
	glm::vec4(255.f, 178.f, 102.f, 255.f) / 255.f,
	glm::vec4(255.f, 255.f, 102.f, 255.f) / 255.f,
	glm::vec4(178.f, 255.f, 102.f, 255.f) / 255.f,
	glm::vec4(102.f, 255.f, 102.f, 255.f) / 255.f,
	glm::vec4(102.f, 255.f, 178.f, 255.f) / 255.f,
	glm::vec4(102.f, 255.f, 255.f, 255.f) / 255.f,
	glm::vec4(102.f, 178.f, 255.f, 255.f) / 255.f,
	glm::vec4(102.f, 102.f, 255.f, 255.f) / 255.f,
	glm::vec4(178.f, 102.f, 255.f, 255.f) / 255.f,
	glm::vec4(255.f, 102.f, 255.f, 255.f) / 255.f,
	glm::vec4(255.f, 102.f, 178.f, 255.f) / 255.f,
	glm::vec4(192.f, 192.f, 192.f, 255.f) / 255.f
};