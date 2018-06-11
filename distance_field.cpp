#include <iostream>
#include <chrono>

#include "include/distance_field.h"

namespace distance_field
{
Point::Point(double dx, double dy) : m_dx(dx), m_dy(dy) { }
double Point::square_distance() { return (std::pow(m_dx, 2.0) + std::pow(m_dy, 2.0)); }

DistanceField::DistanceField(cv::Mat img1, cv::Mat img2):  im_w(img2.cols+1),
                                                           im_h(img2.rows+1),
                                                           INF(calculate_inf()),
                                                           INSIDE(INF, INF),
                                                           EMPTY(0.0, 0.0),
                                                           grid_1(im_w * im_h, INSIDE),
                                                           grid_2(im_w * im_h, INSIDE),
                                                           SDDT1(im_w * im_h, 0.0),
                                                           SDDT2(im_w * im_h, 0.0)
{
  cv::Mat dst1, dst2;
  if(b_sh == 0)
  {
     img1.copyTo(dst1);
     img2.copyTo(dst2);
  }
  else
  {
     cv::copyMakeBorder( img1, dst1, 0, b_sh, 0, b_sh, cv::BORDER_REPLICATE );
     cv::copyMakeBorder( img2, dst2, 0, b_sh, 0, b_sh, cv::BORDER_REPLICATE );
  }

  create_grid(SDDT1, &dst1);
  create_grid(SDDT2, &dst2);
}

DistanceField::DistanceField(cv::Mat img): im_w(img.cols+1),
                                           im_h(img.rows+1),
                                           INF(calculate_inf()),
                                           INSIDE(INF, INF),
                                           EMPTY(0.0, 0.0),
                                           grid_1(im_w * im_h, INSIDE),
                                           grid_2(im_w * im_h, INSIDE),
                                           SDDT1(im_w * im_h, 0.0)
{
    cv::Mat dst;
    if(b_sh == 0)
        img.copyTo(dst);
    else
        cv::copyMakeBorder( img, dst, 0, b_sh, 0, b_sh, cv::BORDER_REPLICATE );

    create_grid(SDDT1, &dst);
}

/*
 * create_grid() - function which preinitialize DDT grid for further calculations
 */
void DistanceField::create_grid(std::vector<double> &grid, cv::Mat *img)
{
  uint r = 0;
  uint g = 0;
  uint b = 0;
  unsigned char *img_buff = static_cast<unsigned char*>(img->data);

    int x, y;

    for (y = 0; y < im_h; y++)
    {
       for (x = 0; x < im_w; x++)
       {
          b = img_buff[x*img->channels()  +y*img->step];
          g = img_buff[x*img->channels()+1+y*img->step];
          r = img_buff[x*img->channels()+2+y*img->step];

          uint max_col = std::max(r, std::max(g, b));

          if(max_col > 0)
          {
             fill_grid_element(grid_1, INSIDE, x, y);
             fill_grid_element(grid_2, EMPTY,  x, y);
          }
          else
          {
             fill_grid_element(grid_1, EMPTY,  x, y);
             fill_grid_element(grid_2, INSIDE, x, y);
          }
       }
    }

  generate_DF(grid_1);
  generate_DF(grid_2);
  merge_grids(grid);
}

/*
 * compare_grid_points(...) - function for comparing values in DDT grid  (8 neighbours algorithm is made)
 */
void DistanceField::compare_grid_points(std::vector<Point> &grid, Point &point, int offsetx, int offsety, int x, int y)
{
  if(x + offsetx >= 0 && x + offsetx < im_w &&
     y + offsety >= 0 && y + offsety < im_h)
  {
    Point other_point = get_grid_element( grid, x + offsetx, y + offsety);
    other_point.m_dx += offsetx;
    other_point.m_dy += offsety;

    if(other_point.square_distance() < point.square_distance())
      point = other_point;
  }
}

/*
 * generate_DF(...) - function which allows one to calculate distance field using 8-point algorithm.
 *                    the central point is chosen and comparing algorithm is called looking in 8 neighbours around
 */
void DistanceField::generate_DF(std::vector<Point> &grid)
{
// Pass 0: firstly filtering grid in forward manner on both x & y, then on x in reverse manner

      int x1, y1;

      for ( y1 = 0; y1 < im_h; y1++)
      {
          for ( x1 = 0; x1 < im_w; x1++)
          {
             Point point = get_grid_element( grid, x1, y1 );
             compare_grid_points( grid, point, -1,  0, x1, y1 );
             compare_grid_points( grid, point,  0, -1, x1, y1 );
             compare_grid_points( grid, point, -1, -1, x1, y1 );
             compare_grid_points( grid, point,  1, -1, x1, y1 );
             fill_grid_element( grid, point, x1, y1 );
           }

           int x2;

           for ( x2 = im_w - 1; x2 >= 0; x2--)
           {
              Point point = get_grid_element( grid, x2, y1);
              compare_grid_points( grid, point, 1, 0, x2, y1);
              fill_grid_element( grid, point, x2, y1);
           }
       }

// Pass 1: firstly filtering grid in reverse manner on both x & y, then on x in forward manner
       int x3, y3;

       for (y3 = im_h - 1; y3 >= 0; y3--)
       {
          for (x3 = im_w - 1; x3 >= 0; x3--)
          {
             Point point = get_grid_element( grid, x3, y3);
             compare_grid_points( grid, point,  1,  0, x3, y3);
             compare_grid_points( grid, point,  0,  1, x3, y3);
             compare_grid_points( grid, point, -1,  1, x3, y3);
             compare_grid_points( grid, point,  1,  1, x3, y3);
             fill_grid_element( grid, point, x3, y3);
          }

          int x4;

          for (x4 = 0; x4 < im_w; x4++)
          {
             Point point = get_grid_element( grid, x4, y3);
             compare_grid_points( grid, point, -1, 0, x4, y3);
             fill_grid_element( grid, point, x4, y3);
          }
       }
}

/*
 * merge_grids(...) - the 8-poiint algorithm generates 2 grids which are finally merged in one with sign defined
 */
void DistanceField::merge_grids(std::vector<double> &grid)
{
    int x, y;

     for( y = 0; y < im_h; y++)
    {
        for( x = 0; x < im_w; x++)
        {
            double dist1 = ( std::sqrt( (double)( get_grid_element( grid_1, x, y ).square_distance() )));
            double dist2 = ( std::sqrt( (double)( get_grid_element( grid_2, x, y ).square_distance() )));
            grid[x + y * im_w] = (dist1 - dist2) * 10.0 / INF;
        }
    }
}

float DistanceField::generate_bilinear_interpolation(float u0, float v0, const std::vector<double> SDDT)
{
   int x = static_cast<int>(u0 * (im_w-1));
   int y = static_cast<int>(v0 * (im_h-1));

   float u = x / im_w;
   float v = y / im_h;

   double u_opp = 1.0f - u;
   double v_opp = 1.0f - v;

   double result = ( SDDT[x+    y*im_w] * u_opp + SDDT[(x+1)+    y*im_w] * u ) * v_opp +
                   ( SDDT[x+(y+1)*im_w] * u_opp + SDDT[(x+1)+(y+1)*im_w] * u ) * v;

   return static_cast<float>(result);
}

#ifdef USE_AFFINE_TRANSFORMATIONS
void DistanceField::inverse_mapping_result(float *res, std::vector<float> SDF, float u_shifted, float v_shifted, cv::Mat affine_m)
{

    cv::invertAffineTransform(affine_m, inv_affine_m);

    float xx = u_shifted * static_cast<float>(im_w-1);
    float yy = v_shifted * static_cast<float>(im_h-1);

    cv::Point3d point = cv::Point3d(xx, yy, 1.0);

    float x = (inv_affine_m.at<double>(0, 0) * point.x +
               inv_affine_m.at<double>(0, 1) * point.y +
               inv_affine_m.at<double>(0, 2) * point.z)+1.0f;

    float y = (inv_affine_m.at<double>(1, 0) * point.x +
               inv_affine_m.at<double>(1, 1) * point.y +
               inv_affine_m.at<double>(1, 2) * point.z)+1.0f;

    float *data = SDF.data();
    float FF1, FF2, aver;

    //SECTION:inside of the defined picture
    if( static_cast<int>(x) < (im_w-1) && static_cast<int>(y) < (im_h-1) && x >= 0.0f && y >= 0.0f )
    {
        *res = get_massive_val(data, static_cast<int>(x), static_cast<int>(y), im_w-1);
    }

    //SECTION: outside of the defined picture, right border
    else if( static_cast<int>(x) >= (im_h-1) && static_cast<int>(y) < (im_h-1) && x >= 0.0f && y >= 0.0f )
    {
        FF1  = get_massive_val(data, im_w-3, static_cast<int>(y), im_w-1);
        FF2  = get_massive_val(data, im_w-2, static_cast<int>(y), im_w-1);

        aver = extrapolate_vals(FF1, FF2, im_w-3, im_w-2, x);
        *res = get_massive_val(data, im_w-2, static_cast<int>(y), im_w-1) + aver;
        if(std::abs(*res) < 2.0f)
          *res *= 2.0f;
    }

    //SECTION: outside of the defined picture, bottom border
    else if( static_cast<int>(x) < (im_w-1) && static_cast<int>(y) >= (im_h-1) && x >= 0.0f && y >= 0.0f )
    {
        FF1  = get_massive_val(data, static_cast<int>(x), im_h-3, im_w-1);
        FF2  = get_massive_val(data, static_cast<int>(x), im_h-2, im_w-1);

        aver = extrapolate_vals(FF1, FF2, im_h-3, im_h-2, static_cast<int>(y));
        *res = get_massive_val(data, static_cast<int>(x), im_h-2, im_w-1) + aver;
        //*res = SDF[static_cast<int>(x) + (im_h-2)*(im_h-1)] - std::sqrt( y*y / ( (im_h-1) * (im_h-1) ));
    }

    //SECTION: outside of the defined picture, top border
    else if( static_cast<int>(x) < (im_w-1) && x >= 0.0f && y < 0.0f )
    {
        FF1  = get_massive_val(data, static_cast<int>(x), 1, im_w-1);
        FF2  = get_massive_val(data, static_cast<int>(x), 0, im_w-1);

        aver = extrapolate_vals(FF1, FF2, 1, 0, static_cast<int>(y));
        *res = get_massive_val(data, static_cast<int>(x), 0, im_w-1) + aver;
    }

    //SECTION: outside of the defined picture, left border
    else if( static_cast<int>(y) < (im_h-1) && x < 0.0f && y >= 0.0f)
    {
        FF1  = get_massive_val(data, 1, static_cast<int>(y), im_w-1);
        FF2  = get_massive_val(data, 0, static_cast<int>(y), im_w-1);

        aver = extrapolate_vals(FF1, FF2, 1, 0, static_cast<int>(x));
        *res = get_massive_val(data, 0, static_cast<int>(y), im_w-1) + aver;
    }

    //SECTION: outside of the defined picture, top right corner
    else if( static_cast<int>(x) >= (im_w-1) && x >= 0.0f && y < 0.0f)
    {
        FF1  = get_massive_val(data, im_w-3, 1, im_w-1);
        FF2  = get_massive_val(data, im_w-2, 0, im_w-1);

        aver = extrapolate_vals(FF1, FF2, 1, 0, static_cast<int>(x));
        *res = get_massive_val(data, im_w-2, 0, im_w-1) + aver;
    }

    //SECTION: outside of the defined picture, bottom right corner
    else if( static_cast<int>(x) >= (im_w-1) && static_cast<int>(y) >= (im_h-1))
    {
        FF1  = get_massive_val(data, im_w-3, im_h-3, im_w-1);
        FF2  = get_massive_val(data, im_w-2, im_h-2, im_w-1);

        aver = extrapolate_vals(FF1, FF2, im_h-3, im_h-2, static_cast<int>(x));
        *res = get_massive_val(data, im_w-2, im_h-2, im_w-1) + aver;
    }

    //SECTION: outside of the defined picture, bottom left corner
    else if ( static_cast<int>(y) >= (im_h-1) && x < 0.0f && y >= 0.0f)
    {
        FF1  = get_massive_val(data, 1, im_h-3, im_w-1);
        FF2  = get_massive_val(data, 0, im_h-2, im_w-1);

        aver = extrapolate_vals(FF1, FF2, im_h-3, im_h-2, static_cast<int>(x));
        *res = get_massive_val(data, 0, im_h-2, im_w-1) + aver;
    }

    //SECTION: outside of the defined picture, top left corner
    else if(x < 0.0f && y < 0.0f)
    {
        FF1  = get_massive_val(data, 1, 1, im_w-1);
        FF2  = get_massive_val(data, 0, 0, im_w-1);

        aver = extrapolate_vals(FF1, FF2, im_w-3, im_w-2, static_cast<int>(x));
        *res = get_massive_val(data, im_w-2, im_h-2, im_w-1) + aver;
    }

    else
    {
        std::cerr << "No conditions in inverse SDF procedure are satisfiied! abort!" << std::endl;
    }
}

#endif

}//namespace distance_field


