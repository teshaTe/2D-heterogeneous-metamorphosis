#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

namespace stb_tricks
{

class StbEnhancements
{
public:
  StbEnhancements(cv::Mat img1, cv::Mat img2);
  StbEnhancements(cv::Mat img1, cv::Mat img2, bool use_rec);
  StbEnhancements() {}
  ~StbEnhancements(){}

  inline std::vector<float> get_circumcircle_radii_1()  { return radius1; }
  inline std::vector<float> get_circumcircle_radii_2()  { return radius2; }
  inline std::vector<cv::Point2f> get_center_coords_1() { return center1; }
  inline std::vector<cv::Point2f> get_center_coords_2() { return center2; }
  inline std::vector<cv::Rect> get_rectangle_box_1()    { return boundRect1; }
  inline std::vector<cv::Rect> get_rectangle_box_2()    { return boundRect2; }
  inline std::vector<std::vector<cv::Point>> get_contours_poly1() { return contours_poly1; }
  inline std::vector<std::vector<cv::Point>> get_contours_poly2() { return contours_poly2; }
  inline int contour_size_1()     { return contours1.size(); }
  inline int contour_size_2()     { return contours2.size(); }
  inline cv::Size thresh_size_1() { return threshold_output1.size(); }
  inline cv::Size thresh_size_2() { return threshold_output2.size(); }

  std::vector<float> get_rectangle_center_coords_1();
  std::vector<float> get_rectangle_center_coords_2();
  std::vector<float> get_uv_shift_for_image2();

  void get_max_circle(float *r1, float *r2, std::vector<float> *c_x, std::vector<float> *c_y);
  cv::Rect get_max_size_rectangle(std::vector<cv::Rect> rec);

  void get_right_step_xy(float *step_x, float *step_y, float shift_x, float shift_y, int total_frames);
  inline void get_rotation_step(double *angle_st, double angle, int total_frames){ *angle_st = angle / static_cast<double>(total_frames); }

  cv::Mat get_affine_transformed_image(cv::Mat *affine_m);
  cv::Mat get_affine_transformed_image(cv::Mat src, cv::Mat *affine_m,  float x_step, float y_step);
  cv::Mat get_affine_transformed_image(cv::Mat src, cv::Mat affine_m);
  cv::Mat get_affine_rotated_image(cv::Mat src, double angle, double scale, cv::Point2f *center);
  inline cv::Mat get_affine_matrix_tr(){ return (cv::Mat_<double>(2 ,3) << 1, 0, get_uv_shift_for_image2()[0],
                                                                           0, 1, get_uv_shift_for_image2()[1]); }

private:
  cv::Mat make_gray_image(cv::Mat src);
  void create_rectangle(int, void*);
  void create_circumcircle(int, void*);

private:
  std::vector<std::vector<cv::Point> > contours_poly1, contours_poly2;
  std::vector<std::vector<cv::Point> > contours1, contours2;
  std::vector<cv::Rect> boundRect1, boundRect2;
  std::vector<cv::Point2f> center1, center2;
  std::vector<float> radius1, radius2;
  std::vector<float> im2_shift;

  cv::Mat threshold_output1, threshold_output2;
  cv::Mat grayImg1, grayImg2;
  cv::Mat image1, image2;

  int thresh = 70; //35
  int max_thresh = 255;
  float total_st_x;
  float total_st_y;
  double total_angle_st;
};

}//namespace stb_tricks
