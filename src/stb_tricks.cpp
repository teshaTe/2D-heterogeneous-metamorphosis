#include "include/stb_tricks.h"

#ifdef USE_OPENMP
  #include <omp.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <iostream>

stb_tricks::StbEnhancements::StbEnhancements(cv::Mat img1, cv::Mat img2)
{
    img1.copyTo(image1);
    img2.copyTo(image2);

    grayImg1 = make_gray_image(image1);
    grayImg2 = make_gray_image(image2);

#if defined(USE_CONE_F3) || defined(USE_EGG_SHAPE_F3)
    create_circumcircle( 0, 0 );
#endif

#ifdef USE_PYRAMID_F3
    create_rectangle( 0, 0 );
#endif

}

stb_tricks::StbEnhancements::StbEnhancements(cv::Mat img1, cv::Mat img2, bool use_rec)
{
    img1.copyTo(image1);
    img2.copyTo(image2);

    grayImg1 = make_gray_image(image1);
    grayImg2 = make_gray_image(image2);

    if(use_rec)
       create_rectangle( 0, 0 );
    else
       create_circumcircle( 0, 0 );
}

std::vector<float> stb_tricks::StbEnhancements::get_rectangle_center_coords_1()
{
    create_rectangle(0, 0);
    std::vector<cv::Rect> rec1 = get_rectangle_box_1();
    cv::Rect max_rec1 = get_max_size_rectangle(rec1);

    float rec1_cx, rec1_cy;
    rec1_cx = max_rec1.x + max_rec1.width  / 2.0;
    rec1_cy = max_rec1.y + max_rec1.height / 2.0;

    std::vector<float> result;
    result.push_back(rec1_cx / image1.cols);
    result.push_back(rec1_cy / image1.rows);

    return result;
}

std::vector<float> stb_tricks::StbEnhancements::get_rectangle_center_coords_2()
{
    create_rectangle(0, 0);
    std::vector<cv::Rect> rec2 = get_rectangle_box_2();
    cv::Rect max_rec2 = get_max_size_rectangle(rec2);

    float rec2_cx, rec2_cy;

    rec2_cx = max_rec2.x + max_rec2.width  / 2.0;
    rec2_cy = max_rec2.y + max_rec2.height / 2.0;

    std::vector<float> result;
    result.push_back(rec2_cx / image2.cols);
    result.push_back(rec2_cy / image2.rows);

    return result;
}

std::vector<float> stb_tricks::StbEnhancements::get_uv_shift_for_image2()
{
    float rec1_cx, rec1_cy, rec2_cx, rec2_cy;
    rec1_cx = get_rectangle_center_coords_1()[0] * image1.cols;
    rec1_cy = get_rectangle_center_coords_1()[1] * image1.rows;

    rec2_cx = get_rectangle_center_coords_2()[0] * image2.cols;
    rec2_cy = get_rectangle_center_coords_2()[1] * image2.rows;

    im2_shift.push_back( rec1_cx - rec2_cx );
    im2_shift.push_back( rec1_cy - rec2_cy );

    return im2_shift;
}

void stb_tricks::StbEnhancements::get_max_circle(float *r1, float *r2, std::vector<float> *cen1, std::vector<float> *cen2)
{
    *r1 = *std::max_element(radius1.begin(), radius1.end());
    *r2 = *std::max_element(radius2.begin(), radius2.end());

    int i1 = std::distance(radius1.begin(), std::find(radius1.begin(), radius1.end(), *r1));
    int i2 = std::distance(radius2.begin(), std::find(radius2.begin(), radius2.end(), *r2));

    cen1->push_back(center1[i1].x / image1.cols);
    cen1->push_back(center1[i1].y / image1.rows);
    cen2->push_back(center2[i2].x / image2.cols);
    cen2->push_back(center2[i2].y / image1.rows);
}

cv::Rect stb_tricks::StbEnhancements::get_max_size_rectangle(std::vector<cv::Rect> rec)
{
    int max_s = rec[0].width * rec[0].height;
    int index = 0;

     for (int i = 0; i < rec.size(); i++)
     {
        if(max_s < rec[i].width * rec[i].height ||
           max_s == rec[i].width * rec[i].height)
        {
           max_s = rec[i].width * rec[i].height;
           index = i;
        }
     }


     return rec[index];
}

cv::Mat stb_tricks::StbEnhancements::get_affine_transformed_image(cv::Mat *affine_m)
{
    cv::Mat dst = cv::Mat::zeros(image2.size(), image2.type());
    *affine_m   = (cv::Mat_<double>(2 ,3) << 1, 0, get_uv_shift_for_image2()[0],
                                             0, 1, get_uv_shift_for_image2()[1]);
    cv::warpAffine(image2, dst, *affine_m, image2.size(), cv::WARP_FILL_OUTLIERS);

    unsigned char *img_buff = static_cast<unsigned char*>(dst.data);

    for( int y = 0; y < dst.rows; ++y)
    {
        for( int x = 0; x < dst.cols; ++x)
        {
            if(img_buff[x*dst.channels()+3+y*dst.step] == 0)
            {
                img_buff[x*dst.channels()  +y*dst.step] = 0;
                img_buff[x*dst.channels()+1+y*dst.step] = 0;
                img_buff[x*dst.channels()+2+y*dst.step] = 0;
                img_buff[x*dst.channels()+3+y*dst.step] = 255;
            }
        }
    }

    return dst;
}

cv::Mat stb_tricks::StbEnhancements::get_affine_transformed_image(cv::Mat src, cv::Mat *affine_m,
                                                                       float x_step, float y_step)
{
    cv::Mat dst = cv::Mat(src.size(), src.type());
    *affine_m = (cv::Mat_<double>(2 ,3) << 1, 0, x_step, 0, 1, y_step);
    cv::warpAffine(src, dst, *affine_m, src.size(), cv::WARP_FILL_OUTLIERS);

    unsigned char *img_buff = static_cast<unsigned char*>(dst.data);

    for(int y = 0; y < dst.rows; ++y)
    {
        for(int x = 0; x < dst.cols; ++x)
        {
            if(img_buff[x*dst.channels()+3+y*dst.step] == 0)
            {
                img_buff[x*dst.channels()  +y*dst.step] = 0;
                img_buff[x*dst.channels()+1+y*dst.step] = 0;
                img_buff[x*dst.channels()+2+y*dst.step] = 0;
                img_buff[x*dst.channels()+3+y*dst.step] = 255;
            }
        }
    }

    return dst;
}

cv::Mat stb_tricks::StbEnhancements::get_affine_transformed_image(cv::Mat src, cv::Mat affine_m)
{
    cv::Mat dst = cv::Mat(src.size(), src.type());
    cv::warpAffine(src, dst, affine_m, src.size(), cv::WARP_FILL_OUTLIERS);

    unsigned char *img_buff = static_cast<unsigned char*>(dst.data);

    for(int y = 0; y < dst.rows; ++y)
    {
        for(int x = 0; x < dst.cols; ++x)
        {
            if(img_buff[x*dst.channels()+3+y*dst.step] == 0)
            {
                img_buff[x*dst.channels()  +y*dst.step] = 0;
                img_buff[x*dst.channels()+1+y*dst.step] = 0;
                img_buff[x*dst.channels()+2+y*dst.step] = 0;
                img_buff[x*dst.channels()+3+y*dst.step] = 255;
            }
        }
    }

    return dst;
}

cv::Mat stb_tricks::StbEnhancements::get_affine_rotated_image(cv::Mat src, double angle, double scale, cv::Point2f *center)
{
    cv::Mat dst = cv::Mat(src.size(), src.type());
    cv::Mat affine_m = cv::getRotationMatrix2D(*center, angle, scale);
    cv::warpAffine(src, dst, affine_m, src.size(), cv::WARP_FILL_OUTLIERS);

    unsigned char *img_buff = static_cast<unsigned char*>(dst.data);

    /*for(int y = 0; y < dst.rows; ++y)
    {
        for(int x = 0; x < dst.cols; ++x)
        {
            if(img_buff[x*dst.channels()+3+y*dst.step] == 0)
            {
                img_buff[x*dst.channels()  +y*dst.step] = 0;
                img_buff[x*dst.channels()+1+y*dst.step] = 0;
                img_buff[x*dst.channels()+2+y*dst.step] = 0;
                img_buff[x*dst.channels()+3+y*dst.step] = 255;
            }
        }
    }*/

    return dst;
}

cv::Mat stb_tricks::StbEnhancements::make_gray_image(cv::Mat src)
{
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    cv::blur( src_gray, src_gray, cv::Size(3,3) );
    return src_gray;
}

void stb_tricks::StbEnhancements::create_circumcircle(int, void *)
{
    std::vector<cv::Vec4i> hierarchy1, hierarchy2;

    /// Detect edges using Threshold
    cv::threshold( grayImg1, threshold_output1, thresh, 255, cv::THRESH_BINARY );
    cv::threshold( grayImg2, threshold_output2, thresh, 255, cv::THRESH_BINARY );

    /// Find contours
    cv::findContours( threshold_output1, contours1, hierarchy1,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    cv::findContours( threshold_output2, contours2, hierarchy2,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    /// Approximate contours to polygons + get bounding rects and circles
    /// First Picture
    contours_poly1.resize(contours1.size());
    center1.resize( contours1.size() );
    radius1.resize( contours1.size() );

    ///Second Picture
    contours_poly2.resize( contours2.size() );
    center2.resize( contours2.size() );
    radius2.resize( contours2.size() );

    for(int i = 0; i < contours1.size(); i++ )
    {
       cv::approxPolyDP( cv::Mat(contours1[i]), contours_poly1[i], 3, true );
       cv::minEnclosingCircle( (cv::Mat)contours_poly1[i], center1[i], radius1[i] );
    }

    for(int j = 0; j < contours2.size(); j++ )
    {
        cv::approxPolyDP( cv::Mat(contours2[j]), contours_poly2[j], 3, true );
        cv::minEnclosingCircle( (cv::Mat)contours_poly2[j], center2[j], radius2[j] );
    }
}

void stb_tricks::StbEnhancements::create_rectangle(int, void*)
{
    std::vector<cv::Vec4i> hierarchy1, hierarchy2;

    /// Detect edges using Threshold
    cv::threshold( grayImg1, threshold_output1, thresh, 255, cv::THRESH_BINARY );
    cv::threshold( grayImg2, threshold_output2, thresh, 255, cv::THRESH_BINARY );

    /// Find contours
    cv::findContours( threshold_output1, contours1, hierarchy1,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    cv::findContours( threshold_output2, contours2, hierarchy2,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    /// Approximate contours to polygons + get bounding rects and circles
    /// First Picture
    contours_poly1.resize(contours1.size());
    boundRect1.resize( contours1.size() );

    ///Second Picture
    contours_poly2.resize(contours2.size());
    boundRect2.resize( contours2.size() );

    for(int i = 0; i < contours1.size(); i++ )
    {
       cv::approxPolyDP( cv::Mat(contours1[i]), contours_poly1[i], 3, true );
       boundRect1[i] = cv::boundingRect( cv::Mat(contours_poly1[i]) );
    }

    for(int j = 0; j < contours2.size(); j++ )
    {
        cv::approxPolyDP( cv::Mat(contours2[j]), contours_poly2[j], 3, true );
        boundRect2[j] = cv::boundingRect( cv::Mat(contours_poly2[j]) );
    }
}

void stb_tricks::StbEnhancements::get_right_step_xy(float *step_x, float *step_y,
                                                    float shift_x, float shift_y,
                                                    int total_frames)
{
    if(shift_x != 0.0f && shift_y != 0.0f)
    {
        *step_x = std::floor(shift_x / total_frames);
        *step_y = std::floor(shift_y / total_frames);
        total_st_x += std::abs(*step_x);
        total_st_y += std::abs(*step_y);
    }
    else if(shift_x != 0.0f && shift_y == 0.0f)
    {
        *step_x = std::floor(shift_x / total_frames);
        *step_y = 0.0f;
        total_st_x += std::abs(*step_x);
    }
    else if(shift_x == 0.0f && shift_y != 0.0f)
    {
        *step_x = 0.0f;
        *step_y = std::floor(shift_y / total_frames);
        total_st_y += std::abs(*step_y);
    }

    if(total_st_x >= std::abs(shift_x) || total_st_y >= std::abs(shift_y))
    {
        *step_x = 0.0f;
        *step_y = 0.0f;
    }
}
