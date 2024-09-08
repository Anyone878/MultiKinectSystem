#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <vector>

#include "../PointCloud.h"

void cuda_BGRA2RGB(std::vector<My::ColorBGRA>& input, std::vector<My::ColorRGB>& output);
void cuda_BGRA2BGR(std::vector<My::ColorBGRA>& input, std::vector<My::ColorBGR>& output);
void cuda_BGR2RGB(std::vector<My::ColorBGR>& input, std::vector<My::ColorRGB>& output);
void cuda_RGB2BGR(std::vector<My::ColorRGB>& input, std::vector<My::ColorBGR>& output);

void cuda_zip_point_color(const std::vector<My::Point>& point_input, const std::vector<My::ColorBGRA>& color_input, std::vector<My::PointColorRGB>& output);
void cuda_unzip_point_color(const std::vector<My::PointColorRGB>& point_colorRGB_input, std::vector<My::Point>& point_output, std::vector<My::ColorBGRA>& colorBGRA_output);
void cuda_unzip_point_color(const std::vector<My::PointColorRGB>& point_colorRGB_input, std::vector<My::Point>& point_output, std::vector<My::ColorRGB>& colorRGB_output);

#endif // !CUDA_UTILS_CUH
