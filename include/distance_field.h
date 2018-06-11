#ifndef DISTANCE_FIELD_H
#define DISTANCE_FIELD_H

#include "include/stb_tricks.h"

#include <ngl/Vec2.h>

#include <vector>
#include <cmath>

namespace distance_field
{

struct Point
{
  Point(double dx, double dy);
  double m_dx, m_dy;
  double square_distance();
};

class DistanceField
{
public:
  DistanceField(cv::Mat img1, cv::Mat img2);
  DistanceField(cv::Mat img);

  ~DistanceField(){ }
  inline std::vector<double> get_SDDT1()         { return SDDT1; }
  inline std::vector<double> get_SDDT2()         { return SDDT2; }

  inline float get_SDDT1_values(cv::Vec2f uv)    { return SDDT1[static_cast<int>((im_w-b_sh) * uv[0]) +
                                                                static_cast<int>((im_h-b_sh) * uv[1]) * im_w]; }

  inline float get_SDDT2_values(cv::Vec2f uv)    { return SDDT2[static_cast<int>((im_w-b_sh) * uv[0]) +
                                                                static_cast<int>((im_h-b_sh) * uv[1]) * im_w]; }

  inline float get_continuous_SDF1(float uv_x, float uv_y) { return generate_bilinear_interpolation(uv_x, uv_y, SDDT1); }
  inline float get_continuous_SDF2(float uv_x, float uv_y) { return generate_bilinear_interpolation(uv_x, uv_y, SDDT2); }

  float generate_bilinear_interpolation(float u0, float v0, const std::vector<double> SDDT);

#ifdef USE_AFFINE_TRANSFORMATIONS
  void inverse_mapping_result(float *res, std::vector<float> SDF, float u_shifted, float v_shifted, cv::Mat affine_m);

private:
  inline float get_massive_val(float *massive, int x, int y, int width) { return massive[x+y*width]; }

  inline float extrapolate_vals(float val1, float val2, int n1, int n2, int cur_n)  { return val1 + (val2 - val1) * (cur_n - n1)/(n2 - n1); }
#endif

private:
  inline int calculate_inf() { return std::max(im_w, im_h) + 1; }
  inline Point get_grid_element(std::vector<Point> &grid, int x, int y) { return grid[x + y * im_w]; }
  inline void fill_grid_element(std::vector<Point> &grid, Point point, int x, int y) { grid[x + y * im_w] = point; }

  void create_grid(std::vector<double> &grid, cv::Mat *img);
  void compare_grid_points(std::vector<Point> &grid, Point &point, int offsetx, int offsety, int x, int y);

  void generate_DF(std::vector<Point> &grid);
  void merge_grids(std::vector<double> &grid);

private:
  cv::Mat inv_affine_m, inv_affine_rot_m;

  int im_w, im_h;
  int b_sh = 1;

  const int INF;
  const Point INSIDE;
  const Point EMPTY;

  std::vector<Point> grid_1;
  std::vector<Point> grid_2;

  std::vector<double> SDDT1, SDDT2;
};

} // namespace distance_field

#endif // DISTANCE_FIELD_H
