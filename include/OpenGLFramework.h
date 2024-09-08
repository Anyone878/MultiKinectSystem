#pragma once

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

class OpenGLFramework {
public:
	// draw pointset FIRST.
	class PointSet {
	public:
		PointSet();
		PointSet(glm::mat4 transformation);

	public:
		void update(int16_t* points, uint8_t* colors, uint32_t num_points);
		void draw(GLuint shader_program);

	private:
		GLuint VAO, VBO, CBO;
		glm::mat4 transformation;
		int16_t* points;
		uint8_t* colors;
		uint32_t num_points;
	};

	class JointSet {
	public:
		class Joint {
		public:
			const glm::vec3 position;
			const glm::vec4 color;
			Joint(glm::vec3 position, glm::vec4 color);
			~Joint();
			void draw_sphere(GLuint shader_program, glm::mat4 transformation, float radius, const int segments = 16, const int rings = 16) const;
		private:
			GLuint VAO, VBO, CBO, EBO;
		};

	public:
		JointSet() : JointSet(glm::mat4(1.0f)) {};
		JointSet(glm::mat4 transformation) : transformation(transformation) {};

	public:
		void add_joint(glm::vec3 position, glm::vec4 color);
		void release_joints();
		void draw(GLuint shader_program, float radius, const int segments = 16, const int rings = 16) const;

	private:
		glm::mat4 transformation;
		std::vector<Joint> joints;
	};

	class BoneSet {
	public:
		class Bone {
		public:
			const glm::vec3 start_position;
			const glm::vec3 end_position;
			const glm::vec4 color;
			Bone(glm::vec3 start, glm::vec3 end, glm::vec4 color);
			~Bone();
			void draw_cylinder(GLuint shader_program, glm::mat4 transformation, float radius, const int segments = 16) const;

		private:
			GLuint VAO, VBO, CBO;
		};
	public:
		BoneSet() : BoneSet(glm::mat4(1.0f)) {};
		BoneSet(glm::mat4 transformation) : transformation(transformation) {};

	public:
		void add_bone(glm::vec3 start, glm::vec3 end, glm::vec4 color);
		void release_bones();
		void draw(GLuint shader_program, float radius, const int segments = 16) const;

	private:
		glm::mat4 transformation;
		std::vector<Bone> bones;
	};

public:
	OpenGLFramework();
	~OpenGLFramework();
	bool init();
	bool windowShouldClose();
	void beforeUpdate();
	void afterUpdate();
	void updatePointCloud(std::vector<PointSet> point_sets);
	void updateJoints(std::vector<JointSet> joint_sets);

private:
	GLFWwindow* window;
	GLuint VAO, VBO, CBO;
	GLuint shaderProgram;
	const int16_t* pointCloud;
	const uint8_t* colors;

	GLuint compileShader(const char* source, GLenum type);
	GLuint createShaderProgram();
	void processInput();
};