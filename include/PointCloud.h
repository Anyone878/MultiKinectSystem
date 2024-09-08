#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

using namespace std;

namespace My {
	struct Point {
		int16_t x, y, z;

		int cal_distance_sq(const Point& p2) const {
			int dx = x - p2.x;
			int dy = y - p2.y;
			int dz = z - p2.z;
			return dx * dx + dy * dy + dz * dz;
		}

		string to_string() const {
			return "Point(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
		}
	};

	struct ColorBGRA {
		uint8_t b, g, r, a;

		string to_string() const {
			return "ColorBGRA(" + std::to_string(b)
				+ ", " + std::to_string(g)
				+ ", " + std::to_string(r)
				+ ", " + std::to_string(a)
				+ ")";
		}
	};

	struct ColorRGB {
		uint8_t r, g, b;

		string to_string() const {
			return "ColorRGB(" + std::to_string(r)
				+ ", " + std::to_string(g)
				+ ", " + std::to_string(b)
				+ ")";
		}
	};

	struct ColorBGR {
		uint8_t b, g, r;

		string to_string() const {
			return "ColorRGB(" + std::to_string(b)
				+ ", " + std::to_string(g)
				+ ", " + std::to_string(r)
				+ ")";
		}
	};

	struct PointColorRGB {
		Point pos;
		ColorRGB color;

		string to_string() const {
			return "PointColorRGB{" + pos.to_string() + " + " + color.to_string() + "}";
		}
	};

	struct PointCloud {
		vector<Point> points;
		vector<ColorBGRA> colors;

		inline size_t kdtree_get_point_count() const { return points.size(); }

		inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
			if (dim == 0) return static_cast<double>(points[idx].x);
			else if (dim == 1) return static_cast<double>(points[idx].y);
			else return static_cast<double>(points[idx].z);
		}

		template <class BBOX>
		bool kdtree_get_bbox(BBOX&) const { return false; }
	};
}