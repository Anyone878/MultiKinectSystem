#include "../include/OpenGLFramework.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstdlib>
#include <spdlog/spdlog.h>

#include "../include/Camera.h"

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

// camera settings
Camera camera(glm::vec3(0.0f, 0.0f, 0.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// Time between current frame and last frame
float deltaTime = 0.0f;
float lastFrame = 0.0f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// shader sources
const char* vertexShaderSource = R"glsl(
#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor; // in BGRA

out vec3 ourColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 transformation;

void main() {
    gl_Position = projection * view * model * transformation * vec4(aPos, 1.0);
    ourColor = aColor.rgb;
	// ourColor = vec3(0.3f, 0.3f, 0.3f);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 460 core
out vec4 FragColor;

in vec3 ourColor;

void main() {
    FragColor = vec4(ourColor, 1.0);
}
)glsl";

OpenGLFramework::OpenGLFramework()
	: window(nullptr), VAO(0), VBO(0), CBO(0), shaderProgram(0) {}

OpenGLFramework::~OpenGLFramework() {
	gladLoaderUnloadGL();
	glfwTerminate();
}

bool OpenGLFramework::init() {
	// Init glfw
	if (!glfwInit()) {
		spdlog::error("Failed to initialize GLFW");
		return false;
	}

	// config glfw
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// create a window
	window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "OpenGL Framework", NULL, NULL);
	if (!window) {
		spdlog::error("Failed to create GLFW window");
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);

	// init glad
	int version = gladLoaderLoadGL();
	spdlog::info("GL {}.{}", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

	// Set callbacks
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glEnable(GL_DEPTH_TEST);

	// build and compile shader program
	shaderProgram = createShaderProgram();
	if (!shaderProgram) {
		spdlog::error("Failed to create shader program");
		return false;
	}

	return true;
}

bool OpenGLFramework::windowShouldClose()
{
	return glfwWindowShouldClose(window);
}

void OpenGLFramework::beforeUpdate()
{
	// per-frame time logic
	float currentFrame = static_cast<float>(glfwGetTime());
	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;

	// Input
	processInput();

	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shaderProgram);

	// update point cloud...
	// and after update
}

void OpenGLFramework::afterUpdate()
{
	// swap buffers and poll IO events
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void OpenGLFramework::updatePointCloud(std::vector<PointSet> point_sets) {
	for (PointSet& set : point_sets) {
		set.draw(shaderProgram);
	}
}

void OpenGLFramework::updateJoints(std::vector<JointSet> joint_sets)
{
	for (JointSet& j : joint_sets) {
		j.draw(shaderProgram, 10.0f);
	}
}

GLuint OpenGLFramework::compileShader(const char* source, GLenum type) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	int success;
	char infoLog[512];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		spdlog::error("ERROR::SHADER::COMPILATION_FAILED {}", infoLog);
	}

	return shader;
}

GLuint OpenGLFramework::createShaderProgram() {
	GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
	GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);

	GLuint program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	int success;
	char infoLog[512];
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		spdlog::error("ERROR::PROGRAM::LINKING_FAILED {}", infoLog);
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return program;
}

void OpenGLFramework::processInput() {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);
}

// Callback functions
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

OpenGLFramework::PointSet::PointSet() : PointSet(glm::mat4(1.0f))
{}

OpenGLFramework::PointSet::PointSet(glm::mat4 transformation) :
	transformation(transformation), points(0), colors(0), num_points(0)
{
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &CBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glVertexAttribPointer(0, 3, GL_SHORT, GL_FALSE, 3 * sizeof(int16_t), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, CBO);
	glVertexAttribPointer(1, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 4 * sizeof(uint8_t), (void*)0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

void OpenGLFramework::PointSet::update(int16_t* points, uint8_t* colors, uint32_t num_points)
{
	this->points = points;
	this->colors = colors;
	this->num_points = num_points;
}

void OpenGLFramework::PointSet::draw(GLuint shader_program)
{
	if (points == nullptr || colors == nullptr)
		throw new std::runtime_error("nullptr when draw pointset.");

	glm::mat4 model = glm::mat4(1.0f);
	model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::scale(model, glm::vec3(0.01f, 0.01f, 0.01f));

	glm::mat4 view = camera.GetViewMatrix();
	glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

	glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "transformation"), 1, GL_FALSE, glm::value_ptr(transformation));

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, num_points * 3 * sizeof(int16_t), points, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, CBO);
	glBufferData(GL_ARRAY_BUFFER, num_points * 4 * sizeof(uint8_t), colors, GL_DYNAMIC_DRAW);

	// point size
	glPointSize(3.0f);

	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, num_points);

	glBindVertexArray(0);
}

void OpenGLFramework::JointSet::add_joint(glm::vec3 position, glm::vec4 color)
{
	joints.emplace_back(position, color);
}

void OpenGLFramework::JointSet::release_joints()
{
	joints.clear();
}

void OpenGLFramework::JointSet::draw(GLuint shader_program, float radius, const int segments, const int rings) const
{
	for (const Joint& j : joints) {
		j.draw_sphere(shader_program, transformation, radius, segments, rings);
	}
}

OpenGLFramework::JointSet::Joint::Joint(glm::vec3 position, glm::vec4 color) : position(position), color(color)
{
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &CBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, CBO);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

OpenGLFramework::JointSet::Joint::~Joint()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &CBO);
	glDeleteBuffers(1, &EBO);
}

void OpenGLFramework::JointSet::Joint::draw_sphere(GLuint shader_program, glm::mat4 transformation, float radius, const int segments, const int rings) const
{
	// create vertices
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec4> colors;
	std::vector<GLuint> indices;
	for (int y = 0; y <= rings; ++y) {
		for (int x = 0; x <= segments; ++x) {
			float xSeg = static_cast<float>(x) / static_cast<float>(segments);
			float ySeg = static_cast<float>(y) / static_cast<float>(rings);
			float xPos = cos(xSeg * 2.0f * glm::pi<float>()) * sin(ySeg * glm::pi<float>());
			float yPos = cos(ySeg * glm::pi<float>());
			float zPos = sin(xSeg * 2.0f * glm::pi<float>()) * sin(ySeg * glm::pi<float>());

			vertices.emplace_back(glm::vec3(xPos, yPos, zPos) * radius + position);
			colors.emplace_back(color);
		}
	}

	for (int y = 0; y < rings; ++y) {
		for (int x = 0; x < segments; ++x) {
			int first = (y * (segments + 1)) + x;
			int second = first + segments + 1;
			indices.emplace_back(first);
			indices.emplace_back(second);
			indices.emplace_back(first + 1);

			indices.emplace_back(second);
			indices.emplace_back(second + 1);
			indices.emplace_back(first + 1);
		}
	}

	// draw the sphere
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, CBO);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec4), colors.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_DYNAMIC_DRAW);

	glBindVertexArray(0);

	glUseProgram(shader_program);
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "transformation"), 1, GL_FALSE, glm::value_ptr(transformation));

	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, GLsizei(indices.size()), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

OpenGLFramework::BoneSet::Bone::Bone(glm::vec3 start, glm::vec3 end, glm::vec4 color) : start_position(start), end_position(end), color(color)
{
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &CBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, CBO);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

OpenGLFramework::BoneSet::Bone::~Bone()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &CBO);
}

void OpenGLFramework::BoneSet::Bone::draw_cylinder(GLuint shader_program, glm::mat4 transformation, float radius, const int segments) const
{
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> colors;
	glm::vec3 axis = glm::normalize(end_position - start_position);
	glm::vec3 temp = (axis != glm::vec3(0, 1, 0)) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
	glm::vec3 normal = glm::normalize(glm::cross(axis, temp));
	glm::vec3 bitangent = glm::cross(axis, normal);

	for (int i = 0; i <= segments; ++i) {
		float angle = static_cast<float>(i) / static_cast<float>(segments) * glm::two_pi<float>();
		glm::vec3 offset = radius * (cos(angle) * normal + sin(angle) * bitangent);
		vertices.emplace_back(start_position + offset);
		vertices.emplace_back(end_position + offset);
		colors.emplace_back(color);
		colors.emplace_back(color);
	}

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, CBO);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec4), colors.data(), GL_DYNAMIC_DRAW);
	glBindVertexArray(0);

	glUseProgram(shader_program);
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "transformation"), 1, GL_FALSE, glm::value_ptr(transformation));

	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, GLsizei(vertices.size()));
	glBindVertexArray(0);
}

void OpenGLFramework::BoneSet::add_bone(glm::vec3 start, glm::vec3 end, glm::vec4 color)
{
	bones.emplace_back(start, end, color);
}

void OpenGLFramework::BoneSet::release_bones()
{
	bones.clear();
}

void OpenGLFramework::BoneSet::draw(GLuint shader_program, float radius, const int segments) const
{
	for (const Bone& b : bones) {
		b.draw_cylinder(shader_program, transformation, radius, segments);
	}
}

