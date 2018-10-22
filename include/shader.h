#ifndef SHADER_H
#define SHADER_H

#include <ngl/Vec4.h>
#include <ngl/Vec2.h>

#include <vector>
#include <memory>

#include "distance_field.h"

namespace shader
{

enum class DrawMode
{
  WHITE,
  BLOCK_COLOR,
  SIMPLE_BLEND,
  SIMPLE_CLOSEST
};

enum class ColorMode
{
  RGB,
  HSV,
  CIELab,
  CMYK
};

class ColorShader
{

public:
  ColorShader(int length, cv::Mat img1, cv::Mat img2, ColorMode color_mode, DrawMode draw_mode);
  ~ColorShader() {}

  void color_image(ngl::Vec4 *color, ngl::Vec2 coord, int frame);

  inline void update_image2(cv::Mat new_img2) { image2.empty(); new_img2.copyTo(image2); }
  inline void update_im2_buff() {im2_buf = 0; im2_buf = static_cast<unsigned char*>(image2.data); }
  inline cv::Mat get_image2() { return image2; }

  inline std::vector<float> get_f2_values(){ return F2; }
  inline void update_f2_tr_values(std::vector<float> new_f2) { F2_m.clear(); F2_m = new_f2; }

  inline std::vector<float> get_rec1_center(){ return center_r1; }
  inline std::vector<float> get_rec2_center(){ return center_r2; }

  std::shared_ptr<distance_field::DistanceField> DF;

#ifdef USE_AFFINE_TRANSFORMATIONS
  inline void set_inverse_affine_matrix(cv::Mat affine_m){ cv::invertAffineTransform(affine_m, inv_affine_m); }
  inline void update_ai_coeffs(){ set_ai_coeffs(); }
  inline cv::Mat get_affine_matrix() { return affine_m; }
  inline void update_affine_matrix(cv::Mat affine) { affine_m.at<double>(0,2) = affine_m.at<double>(0,2) + affine.at<double>(0,2);
                                                     affine_m.at<double>(1,2) = affine_m.at<double>(1,2) + affine.at<double>(1,2); }
#endif

  //functions for calculating STB and STTI opearions
private:
  ngl::Vec4 obtain_color(unsigned char *img_buf,cv::Mat *img, ngl::Vec2 uv);
  ngl::Vec4 BB_shade_surface(ngl::Vec2 uv, float time);
  ngl::Vec4 final_color(ngl::Vec2 uv, float f1, float f2, float fun1, float fun2);

  float smooth_cylynder1(float time, float fun);
  float smooth_cylynder2(float time, float fun);

  void blend_colors(ngl::Vec2 uv, ngl::Vec4 *col1, ngl::Vec4 *col2, float f1, float f2);
  void block_color(ngl::Vec2 uv, ngl::Vec4 *col1, ngl::Vec4 *col2);
  void closest_color(ngl::Vec2 uv, ngl::Vec4 *col1, ngl::Vec4 *col2);
  void clamp_color(ngl::Vec4 *col);

  float function3(ngl::Vec2 uv, float time);

  ngl::Vec4 cross_dissolve(ngl::Vec4 col1, ngl::Vec4 col2, float time);

  //functions for ai coeffs interval estimation
private:
  float calc_a1_coeff();
  std::vector<float> calc_a0_interval();
  void set_ai_coeffs();
  bool check_2circle_crosssection();

  //functions for calculating 3 operations on R-functions and min/max for 3 and 4 variables
private:
  inline float intersect_function(float a, float b) { return a + b - std::sqrt(a * a + b * b); }
  inline float union_function(float a, float b)     { return a + b + std::sqrt(a * a + b * b); }
  inline float subtract_function(float a, float b)  { return intersect_function(a, -b); }

  inline float min3(float a, float b, float c)      { return std::min(std::min(a, b), c); }
  inline float max3(float a, float b, float c)      { return std::max(std::max(a, b), c); }

  inline float min4(float a, float b, float c, float d) { return std::min(a, std::min(b, std::min(c,d))); }
  inline float max4(float a, float b, float c, float d) { return std::max(a, std::max(b, std::max(c,d))); }

  //functions for conducting operations on colour conversion between colour spaces
private:
  void convert_from_RGB(float *r, float *g, float *b);
  void convert_to_RGB(float *r, float *g, float *b);
  void convert_RGB_to_HSV(float *r, float *g, float *b);
  void convert_HSV_to_RGB(float *r, float *g, float *b);
  void convert_RGB_to_CIELab(float *r, float *g, float *b);
  void convert_CIELab_to_RGB(float *r, float *g, float *b);
  //void convert_RGB_to_CMY(float *r, float *g, float *b);
  //void convert_CMY_to_RGB(float *r, float *g, float *b);

  //image variables
private:
  DrawMode draw_mode_name;
  ColorMode color_mode_name;

  cv::Mat image1;
  cv::Mat image2;
  cv::Mat affine_m, inv_affine_m, rot_affine_m;
  unsigned char *im1_buf, *im2_buf;

  //STB and general variables
private:
  int resolution_x, resolution_y;
  int block_size_param;
  int length_param;

  float a0,a1,a2,a3;
  std::vector<float> F1, F2, F2_m;

  //circumscribed circles around shapes
private:
  float rad1, rad2, r1, r2;
  float rec1_w, rec1_h, rec2_w, rec2_h;

  std::vector<float> center1,   center2;
  std::vector<float> center_r1, center_r2;

  cv::Rect rec1, rec2;

  //private paramters for a0 and a3 interval estimation
private:
  float x_min, x_max;
  float y_min, y_max;
  //float t_min, t_max;
  float A1, B1, C1;
  float A2, B2, C2;
  float cx1, cy1, cx2, cy2;
  float r1_uv, r2_uv;

  std::vector<float> fun1, fun2;
  std::vector<float> fun11, fun22;
  std::vector<float> f11f22_sum, SqrF12F22_sum, f1f2_sum;
  std::vector<float> a0_int;

  //parameters for smoothing cylinders
private:
  float max_a3 = 1.0f;

  //1st cylinder
  float f3_pyr_1  = -0.3f;
  float f3_cone_1 = -0.8f; //-0.8

#if not defined(USE_CONE_F3) || defined(USE_PYRAMID_F3)
  float b1_0 = -0.3f;       //-0.3f
#else
  float b1_0;
#endif
  float b1_1 = 1.0;
  float b1_2 = 1.0;
  float b1_3 = 1.0;

  //2nd cylinder
  float f3_pyr_2  = -0.5f;
  float f3_cone_2 = -0.2f; //-0.5

#if not defined(USE_CONE_F3) || defined(USE_PYRAMID_F3)
  float b2_0 = -0.5f;       //-0.5f
#else
  float b2_0;
#endif

  float b2_1 = 1.0;
  float b2_2 = 1.0;
  float b2_3 = 1.0;
};

}//namespace shader

#endif // SHADER_H
