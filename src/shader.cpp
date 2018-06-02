#ifdef USE_OPENMP
  #include <omp.h>
#endif

#include <cmath>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <ctime>
#include <ratio>

#include "include/shader.h"

#define UNION shader::ColorShader::union_function
#define INTERSECT shader::ColorShader::intersect_function
#define SUBTRACT shader::ColorShader::subtract_function

shader::ColorShader::ColorShader(int length, int block_size, cv::Mat img1, cv::Mat img2,
                                 ColorMode color_mode, DrawMode draw_mode)
{
    length_param     = length;
    draw_mode_name   = draw_mode;
    color_mode_name  = color_mode;
    block_size_param = block_size;

    img1.copyTo(image1);
    img2.copyTo(image2);
    im1_buf = static_cast<unsigned char*>(image1.data);
    im2_buf = static_cast<unsigned char*>(image2.data);
    resolution_x = image1.cols;
    resolution_y = image1.rows;

    DF = nullptr;
    set_ai_coeffs();

#ifdef USE_DISTANCE_FIELD

    auto start = std::chrono::high_resolution_clock::now();

    DF = std::make_shared<distance_field::DistanceField>(image1, image2);
    float x1, y1;

    for(int i1= 0; i1 < image1.rows; ++i1)
    {
        for(int j1 = 0; j1 < image1.cols; ++j1)
        {
            x1 = static_cast<float>(j1) / static_cast<float>(resolution_x);
            y1 = static_cast<float>(i1) / static_cast<float>(resolution_y);
            F1.push_back(DF.get()->get_continuous_SDF1(x1, y1));
        }
    }

    float x2, y2;

    for(int i2 = 0; i2 < resolution_y; ++i2)
    {
        for(int j2 = 0; j2 < resolution_x; ++j2)
        {
            x2 = static_cast<float>(j2) / static_cast<float>(resolution_x);
            y2 = static_cast<float>(i2) / static_cast<float>(resolution_y);
            F2.push_back(DF.get()->get_continuous_SDF2(x2, y2));
        }
    }

#ifndef USE_AFFINE_TRANSFORMATIONS
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Preproc. time (around distance_fields): " << dur.count() << " seconds" << std::endl;
#endif

#endif // USE_DISTANCE_FIELD

#if defined(USE_AFFINE_TRANSFORMATIONS) || defined(USE_CONE_F3) || defined(USE_PYRAMID_F3) || defined(USE_EGG_SHAPE_F3)

#if defined(USE_CONE_F3) || defined(USE_EGG_SHAPE_F3)
    stb_tricks::StbEnhancements stb(image1, image2);
    stb.get_max_circle(&rad1, &rad2, &center1, &center2);
#endif //USE_CONE_F3

#ifdef USE_PYRAMID_F3
    stb_tricks::StbEnhancements stb(image1, image2);
    rec1 = stb.get_max_size_rectangle(stb.get_rectangle_box_1());
    rec2 = stb.get_max_size_rectangle(stb.get_rectangle_box_2());
    center_r1 = stb.get_rectangle_center_coords_1();
    center_r2 = stb.get_rectangle_center_coords_2();
#endif //USE_PYRAMID_F3

#ifdef USE_AFFINE_TRANSFORMATIONS
    stb_tricks::StbEnhancements stb(image1, image2, true);
    image2    = stb.get_affine_transformed_image(&affine_m);
    center_r1 = stb.get_rectangle_center_coords_1();
    center_r2 = stb.get_rectangle_center_coords_2();

    im2_buf = static_cast<unsigned char*>(image2.data);
    float res = 0.0f;

    for(int y = 0; y < image2.rows; ++y)
    {
        for(int x = 0; x < image2.cols; ++x)
        {
            float u_x = static_cast<float>(x)/static_cast<float>(image2.rows);
            float u_y = static_cast<float>(y)/static_cast<float>(image2.cols);

            DF.get()->inverse_mapping_result(&res, F2, u_x, u_y, affine_m);
            F2_m.push_back(res);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Preproc. time (around distance_fields): " << dur.count() << " seconds" << std::endl;
#endif //USE_AFFINE_TRANSFORMATIONS
#endif //defined(USE_AFFINE_TRANSFORMATIONS) || defined(USE_CONE_F3) || defined(USE_PYRAMID_F3)
}

void shader::ColorShader::set_ai_coeffs()
{

    std::vector<float> cen1, cen2;
    float r1, r2;

    stb_tricks::StbEnhancements stb(image1, image2, false);
    stb.get_max_circle(&r1, &r2, &cen1, &cen2);
    float dist = std::sqrt(std::pow((cen2[0]*static_cast<float>(resolution_x) - cen1[0]*static_cast<float>(resolution_x)), 2.0) +
                           std::pow((cen2[1]*static_cast<float>(resolution_y) - cen1[1]*static_cast<float>(resolution_y)), 2.0));

#if defined (USE_CONE_F3) || defined(USE_PYRAMID_F3)
#ifdef USE_PYRAMID_F3
    a1 = std::exp(dist / (2.0f*(r1 + r2)));
    a2 = std::exp(dist / (2.0f*(r1 + r2)));
    a3 = 0.017;
    a0 = 2.5;
#else
    a2 = std::exp(dist / (2.0f * (r1 + r2)));
    a1 = 0.7*a2;
    a3 = 0.004;
    a0 = 5.0;
#endif
#else

    a1 = std::exp(dist / (2.0f*(r1 + r2)));
    a2 = std::exp(dist / (2.0f*(r1 + r2)));
    a3 = a1*std::exp(-0.65f*std::exp(-0.7f*a1));

    if(a3 < 1.0f)
      a3 = 1.0f;

    //for understanding what is going on here look at the picture
    //yA is the y coordinate of A1' point from the picture
    //xB is the x coordinate of B1' point from the picture
    float yA = 20.0f - std::sqrt(std::pow(a1+10.0f, 2.0f) + std::pow(10.0f-a1, 2.0f));
    float xB = std::sqrt(0.5f * ( std::pow(20.0f-a1, 2.0f) - 200.0f ) );

    //xCC, yCC - coordinates of intersection of the A1'B1' and OC lines
    float k1  = xB * yA - a2 * a1;
    float k2  = xB - a2 - a1 + yA;

    float yCC = k1 / k2;
    float xCC = yCC;

    a0 = std::sqrt(yCC*yCC + xCC*xCC);    
#endif
}

/*
 * obtain_color() - function, which obtaining the color of the pixel from the loaded picture;
 */
ngl::Vec4 shader::ColorShader::obtain_color(unsigned char *img_buf,cv::Mat *img, ngl::Vec2 uv)
{
    uint r = 0;
    uint g = 0;
    uint b = 0;
    uint a = 0;

    int x = static_cast<int>(uv.m_x * static_cast<float>(resolution_x)) * img->channels();
    int y = static_cast<int>(uv.m_y * static_cast<float>(resolution_y));

    b = img_buf[x+  y*img->step];
    g = img_buf[x+1+y*img->step];
    r = img_buf[x+2+y*img->step];
    a = img_buf[x+3+y*img->step];

    ngl::Vec4 col( static_cast<float>(r/255.0f), static_cast<float>(g/255.0f),
                   static_cast<float>(b/255.0f), static_cast<float>(a/255.0f) );

    convert_from_RGB(&col.m_r, &col.m_g, &col.m_b);
    return col;
}

/*
 * color_image() - function, which received color for the pixel after applying belnding operation
 *                 between two images;
 */
void shader::ColorShader::color_image(ngl::Vec4 *color, ngl::Vec2 coord, int frame)
{
    float time = 0.0;
    ngl::Vec2 uv = ngl::Vec2(coord.m_x / static_cast<float>(resolution_x),
                             coord.m_y / static_cast<float>(resolution_y));

    float t = static_cast<float>(frame) / static_cast<float>(length_param);
    if(t < 0.01f || t == 0.01f)
    {
       time = -10.0f;
       *color = BB_shade_surface(uv, time);
    }
    else
    {
       time = ( 20.0f / 0.36f ) * ( t - 0.01f ) * ( t - 0.01f ) - 10.0f;
       *color = BB_shade_surface(uv, time);
    }
}

/*
 * BB_shade_surface() - function, which conducts space-time blending operation between two forms for the geometry;
 *                      on the basis of the result it calls obtain_color() or final_color() for obtaining the color
 *                      value of the current pixel;
 */
ngl::Vec4 shader::ColorShader::BB_shade_surface(ngl::Vec2 uv, float time)
{
  float fun1, fun2;
  float f1, f2;

#ifdef USE_SMOOTHED_CYLINDERS
  fun1 = function1(uv.m_x, uv.m_y);
  fun2 = function2(uv.m_x, uv.m_y);

  f1 = smooth_cylynder1(time, fun1);
  f2 = smooth_cylynder2(time, fun2);
#else
  fun1 = function1(uv.m_x, uv.m_y);
  fun2 = function2(uv.m_x, uv.m_y);

  f1 = INTERSECT(fun1, -time);
  f2 = INTERSECT(fun2, (time - 1.0f));
#endif //USE_SMOOTHED_CYLINDERS

  float f3 = function3(uv, time);

  float r1 = std::pow(f1/a1, 2.0f) + std::pow(f2/a2, 2.0f);
  float r2 = 0.0f;

  if( f3 > 0.0f )
  {
    r2 = std::pow(f3/a3, 2.0f);
  }

  float rr = 0.0f;
  if( r1 > 0.0f )
  {
    rr = r1 / (r1 + r2);
  }

  float d = 0.0f;
  if( rr < 1.0f )
  {
    d = a0 * std::pow(1.0f - rr, 3.0f) / (1.0f + rr);
  }

  float blending_result = UNION(f1, f2) + d;

  ngl::Vec4 surf_col(0.0, 0.0, 0.0, 1.0);
  if(blending_result >= 0.0f)
  {
    if(f1 < 0.0f && f2 < 0.0f)
    {
      surf_col = final_color(uv, f1, f2, fun1, fun2);
    }
    else
    {
       if(f1 >= 0.0f)
       {
         surf_col = obtain_color(im1_buf, &image1, uv);
       }

       if(f2 >= 0.0f)
       {
         surf_col = obtain_color(im2_buf, &image2, uv);
       }
    }
  }

  convert_to_RGB(&surf_col.m_r, &surf_col.m_g, &surf_col.m_b);
  clamp_color(&surf_col);

#ifdef USE_TRANSPARENT_BACKBGROUND
  if(surf_col == ngl::Vec4(0.0, 0.0, 0.0, 1.0))
    surf_col = ngl::Vec4(0.0, 0.0, 0.0, 0.0);
#endif

  return surf_col;
}

double shader::ColorShader::smooth_cylynder1(float time, float fun)
{
#if defined (USE_PYRAMID_F3) || defined(USE_CONE_F3)
#ifdef USE_PYRAMID_F3
    float b0 = -0.3;
#else
    float b0 = -0.8;
#endif
#else
    float b0 = -0.3;
#endif
    float b1 = 1.0;
    float b2 = 1.0;
    float b3 = 1.0;

    float f1 = fun;
    float f2 = -time;
    float f3 =  time + 5.0f;

    float r1 = std::pow(f1/b1, 2.0f) + std::pow(f2/b2, 2.0f);
    float r2 = 0.0f;

    if( f3 > 0.0f )
    {
      r2 = std::pow(f3/b3, 2.0f);
    }

    float rr = 0.0f;
    if( r1 > 0.0f )
    {
      rr = r1 / (r1 + r2);
    }

    float d = 0.0f;
    if( rr < 1.0f )
    {
      d = b0 * std::pow(1.0f - rr, 3.0f) / (1.0f + rr);
    }

    float blending_result = INTERSECT(f1, f2) + d;
    return blending_result;
}

double shader::ColorShader::smooth_cylynder2(float time, float fun)
{
#if defined (USE_PYRAMID_F3) || defined(USE_CONE_F3)
#ifdef USE_PYRAMID_F3
    float b0 = -0.5;
#else
    float b0 = -0.5;
#endif
#else
    float b0 = -0.5;
#endif

    float b1 = 1.0;
    float b2 = 1.0;
    float b3 = 1.0;

    float f1 = fun;
    float f2 = time - 1.0f;
    float f3 = 5.0f - time;

    float r1 = std::pow(f1/b1, 2.0f) + std::pow(f2/b2, 2.0f);
    float r2 = 0.0f;

    if( f3 > 0.0f )
    {
      r2 = std::pow(f3/b3, 2.0f);
    }

    float rr = 0.0f;
    if( r1 > 0.0f )
    {
      rr = r1 / (r1 + r2);
    }

    float d = 0.0f;
    if( rr < 1.0f )
    {
      d = b0 * std::pow(1.0f - rr, 3.0f) / (1.0f + rr);
    }

    float blending_result = INTERSECT(f1, f2) + d;
    return blending_result;
}

void shader::ColorShader::blend_colors(ngl::Vec2 uv, ngl::Vec4 *col1, ngl::Vec4 *col2, float f1, float f2)
{
  float result1  = f1;
  float result2  = f2;
  float u_x = uv.m_x;
  float u_y = uv.m_y;

  if(result1 < 0.0f)
  {
      float total_w1 = 0.0f;
      float r1 = 0;
      float g1 = 0;
      float b1 = 0;

#ifdef USE_OPENMP
#pragma omp target teams distribute parallel for collapse(2) map(from: r1,g1,b1, total_w1) reduction(+:total_w1,r1,g1,b1) map(to: u_x,u_y)
#endif
     for(int i1 = 0; i1 < resolution_y; ++i1)
     {
        for(int j1 = 0; j1 < resolution_x; ++j1)
        {
           float x1 = static_cast<float>(j1) / static_cast<float>(resolution_x);
           float y1 = static_cast<float>(i1) / static_cast<float>(resolution_y);

           if(function1(x1, y1) >= 0.0f)
           {
               float b = im1_buf[j1*image1.channels()+  i1*image1.step]/255.0f;
               float g = im1_buf[j1*image1.channels()+1+i1*image1.step]/255.0f;
               float r = im1_buf[j1*image1.channels()+2+i1*image1.step]/255.0f;
               convert_from_RGB(&r, &g, &b);

               b1 += b / ((std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f)) *
                          (std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f)));

               g1 += g / ((std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f)) *
                          (std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f)));

               r1 += r / ((std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f)) *
                          (std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f)));

               total_w1 += 1.0f / ((std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f)) *
                                   (std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f)));
           }
        }
     }

     *col1 = ngl::Vec4(r1 / total_w1, g1 / total_w1, b1 / total_w1, 1.0f);
  }

  if(result2 < 0.0f)
  {
      float total_w2 = 0.0f;
      float r2 = 0;
      float g2 = 0;
      float b2 = 0;

#ifdef USE_OPENMP
#pragma omp target teams distribute parallel for collapse(2) map(from:r2,g2,b2, total_w2) reduction(+:total_w2,r2,g2,b2) map(to: u_x, u_y)
#endif
     for(int i2 = 0; i2 < resolution_y; ++i2)
     {
        for(int j2 = 0; j2 < resolution_x; ++j2)
        {
           float x2 = static_cast<float>(j2) / static_cast<float>(resolution_x);
           float y2 = static_cast<float>(i2) / static_cast<float>(resolution_y);

           if(function2(x2, y2) >= 0.0f)
           {
               float b = im2_buf[j2*image2.channels()+  i2*image2.step]/255.0f;
               float g = im2_buf[j2*image2.channels()+1+i2*image2.step]/255.0f;
               float r = im2_buf[j2*image2.channels()+2+i2*image2.step]/255.0f;
               convert_from_RGB(&r, &g, &b);

               b2 += b / ((std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f)) *
                          (std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f)));

               g2 += g / ((std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f)) *
                          (std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f)));

               r2 += r / ((std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f)) *
                          (std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f)));

               total_w2 += 1.0f / ((std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f)) *
                                   (std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f)));
           }
        }
     }

     *col2 = ngl::Vec4(r2 / total_w2, g2 / total_w2, b2 / total_w2, 1.0f);
  }
}

/*
 * block_color() - function, which obtain color of the pixel for both 1 picture and final picture
 *                 between which blending operation is made;
 */
void shader::ColorShader::block_color(ngl::Vec2 uv, ngl::Vec4 *col1, ngl::Vec4 *col2)
{
  *col1 = obtain_color(im1_buf, &image1, uv);
  *col2 = obtain_color(im2_buf, &image2, uv);
}

/*
 * closest_color() - function, which set undefined pixel colors to the closest pixel defined by the input texture;
 *                   working with 2 input images color;
 */
void shader::ColorShader::closest_color(ngl::Vec2 uv, ngl::Vec4 *col1, ngl::Vec4 *col2)
{
  ngl::Vec2 closest_uv1;
  ngl::Vec2 closest_uv2;
  float u_x = uv.m_x;
  float u_y = uv.m_y;
  float cl1_uv_x, cl1_uv_y;

#ifdef USE_OPENMP
#pragma omp target teams distribute parallel for collapse(2) map(from: cl1_uv_x, cl1_uv_y) map(to: u_x, u_y)
#endif
  for(int i1 = 0; i1 < resolution_y; ++i1)
  {
      for(int j1 = 0; j1 < resolution_x; ++j1)
      {
         float x1 = static_cast<float>(j1)/static_cast<float>(resolution_x);
         float y1 = static_cast<float>(i1)/static_cast<float>(resolution_x);
         float dist_squared = std::pow(u_x - x1, 2.0f) + std::pow(u_y - y1, 2.0f);

         if(function1(x1, y1) >= 0.0f && dist_squared < smallest_dist_squared1)
         {
            smallest_dist_squared1 = dist_squared;
            cl1_uv_x = x1;
            cl1_uv_y = y1;
         }
      }
  }

  float cl2_uv_x, cl2_uv_y;

#ifdef USE_OPENMP
#pragma omp target teams distribute parallel for collapse(2) map(from: cl2_uv_x, cl2_uv_y) map(to: u_x, u_y)
#endif
  for(int i2 = 0; i2 < resolution_y; ++i2)
  {
      for(int j2 = 0; j2 < resolution_x; ++j2)
      {
        float x2 =  static_cast<float>(j2)/static_cast<float>(resolution_x);
        float y2 =  static_cast<float>(i2)/static_cast<float>(resolution_x);
        float dist_squared = std::pow(u_x - x2, 2.0f) + std::pow(u_y - y2, 2.0f);

        if(function2(x2, y2) >= 0.0f && dist_squared < smallest_dist_squared2)
        {
           smallest_dist_squared2 = dist_squared;
           cl2_uv_x = x2;
           cl2_uv_y = y2;
        }
     }
  }

  closest_uv1 = ngl::Vec2(cl1_uv_x, cl1_uv_y);
  closest_uv2 = ngl::Vec2(cl2_uv_x, cl2_uv_y);

  *col1 = obtain_color(im1_buf, &image1, closest_uv1);
  *col2 = obtain_color(im2_buf, &image2, closest_uv2);
}

/*
 * clamp_color() - function, which clamp the color if it`s pixel values are less the 0 or more then 1;
 */
void shader::ColorShader::clamp_color(ngl::Vec4 *col)
{
  if(col->m_r <= 0.0f) col->m_r = 0.0f;
  if(col->m_g <= 0.0f) col->m_g = 0.0f;
  if(col->m_b <= 0.0f) col->m_b = 0.0f;

  if(col->m_r >= 1.0f) col->m_r = 1.0f;
  if(col->m_g >= 1.0f) col->m_g = 1.0f;
  if(col->m_b >= 1.0f) col->m_b = 1.0f;
}

/*
 * final_color() - function, which calculates the final color of the pixel while color interpolation
 *                 procedure is conducted;
 */
ngl::Vec4 shader::ColorShader::final_color(ngl::Vec2 uv, float f1, float f2, float fun1, float fun2)
{
  ngl::Vec4 surf_col;
  ngl::Vec4 col1 = obtain_color(im1_buf, &image1, uv);
  ngl::Vec4 col2 = obtain_color(im2_buf, &image2, uv);
  ngl::Vec4 zero(0.0, 0.0, 0.0, 1.0);

  if(col1 == zero || col2 == zero)
  {
    switch (draw_mode_name)
    {
      case DrawMode::WHITE:
        if(col1 == zero) col1 = ngl::Vec4(1.0, 1.0, 1.0, 1.0);
        if(col2 == zero) col2 = ngl::Vec4(1.0, 1.0, 1.0, 1.0);
        break;
      case DrawMode::BLOCK_COLOR:
        block_color(uv, &col1, &col2);
        break;
      case DrawMode::SIMPLE_BLEND:
        blend_colors(uv, &col1, &col2, fun1, fun2);
        break;
      case DrawMode::SIMPLE_CLOSEST:
        closest_color(uv, &col1, &col2);
        break;
      default:
        break;
    }
  }

  float fsum = std::fabs(f1) + std::fabs(f2);
  float w1   = std::fabs(f2) / fsum;
  float w2   = std::fabs(f1) / fsum;

  surf_col.m_x = col1.m_x * w1 + col2.m_x * w2;
  surf_col.m_y = col1.m_y * w1 + col2.m_y * w2;
  surf_col.m_z = col1.m_z * w1 + col2.m_z * w2;
  surf_col.m_a = 1.0f;

  return surf_col;
}

/*
 * function1() - controlls the shape of the 1 object
 */
float shader::ColorShader::function1(float uv_x, float uv_y)
{
   float result = 0.0f;

#ifdef USE_DISTANCE_FIELD
   result = F1[static_cast<int>(uv_x*resolution_x) + static_cast<int>(uv_y*resolution_y*resolution_x)];
#endif // defined(USE_DISTANCE_FIELD)

#ifdef USE_CIRCLE_CIRCLE_CROSS
  uv_y = 1.0f - uv_y;
  ngl::Vec2 pos = ngl::Vec2(uv_x * 10.0f - 5.0f, uv_y * 10.0f - 3.0f);

  float disk1 = 1.0f - (pos.m_x - 1.0f) * (pos.m_x - 1.0f) - pos.m_y * pos.m_y;
  float disk2 = 1.0f - (pos.m_x + 1.5f) * (pos.m_x + 1.5f) - pos.m_y * pos.m_y;
  float ddisk = UNION(disk1, disk2);
  result = ddisk;
#endif //TWO_CIRCLES_CROSS

#ifdef USE_CIRCLE_RING_CROSS
  uv_y = 1.0f - uv_y;
  ngl::Vec2 pos = ngl::Vec2(uv_x * 10.0f - 5.0f, uv_y * 10.0f - 3.0f);

  float disk1    = 1.0f - (pos.m_x - 1.0f) * (pos.m_x - 1.0f) - pos.m_y * pos.m_y;

  float disk_out = 1.0f - (pos.m_x + 1.5f) * (pos.m_x + 1.5f) - pos.m_y * pos.m_y;
  float disk_in  = 0.5f - (pos.m_x + 1.5f) * (pos.m_x + 1.5f) - pos.m_y * pos.m_y;
  float ring     = SUBTRACT(disk_out, disk_in);

  float ddisk = UNION(disk1, ring);
  result = ddisk;
#endif // CIRCLE_RING_CROSS

#ifdef USE_CIRCLE_CIRCLE
  uv.m_y = 1.0f - uv_y;
  ngl::Vec2 pos = ngl::Vec2(uv_x * 10.0f - 5.0f, uv_y * 10.0f - 3.0f);

  result = 3.0f - pos.m_x * pos.m_x - pos.m_y * pos.m_y;
#endif //CIRCLE_CIRCLE

  return result;
}


/*
 * function2() - controlls the shape of the second object
 */
float shader::ColorShader::function2(float uv_x, float uv_y)
{
    float result = 0.0f;

#ifdef USE_DISTANCE_FIELD
#ifndef USE_AFFINE_TRANSFORMATIONS
    result = F2[static_cast<int>(uv_x*resolution_x) + static_cast<int>(uv_y*resolution_y*resolution_x)];
#else
    result = F2_m[static_cast<int>(uv_x*resolution_x) + static_cast<int>(uv_y*resolution_y*resolution_x)];
#endif
#endif// defined(USE_DISTANCE_FIELD) && not defined(USE_AFFINE_TRANSFORMATIONS)

#if defined (USE_CIRCLE_CIRCLE_CROSS) || defined (USE_CIRCLE_RING_CROSS)
    uv_y = 1.0f - uv_y;
    ngl::Vec2 pos = ngl::Vec2(uv_x * 10.0f - 5.0f, uv_y * 10.0f - 3.0f);

    float bl1 = INTERSECT(INTERSECT(INTERSECT((pos.m_x + 1.0f),  (0.0f - pos.m_x)), 
                                              (pos.m_y - 2.0f)), (5.0f - pos.m_y));
    float bl2 = INTERSECT(INTERSECT(INTERSECT((pos.m_y - 3.0f),  (4.0f - pos.m_y)), 
                                              (pos.m_x + 2.0f)), (1.0f - pos.m_x));
    float cross = UNION(bl1, bl2);
    result = cross;

#endif //defined (TWO_CIRCLES_CROSS) || defined (CIRCLE_RING_CROSS)

#ifdef USE_CIRCLE_CIRCLE
  uv.m_y = 1.0f - uv_y;
  ngl::Vec2 pos = ngl::Vec2(uv_x * 10.0f - 5.0f, uv_y * 10.0f - 6.0f);

  result = 0.6f - pos.m_x * pos.m_x - pos.m_y * pos.m_y;

#endif //CIRCLE_CIRCLE

  return result;
}

/*
 * function3() - controlls the bounding box insied of which space-time blending took place
 */
float shader::ColorShader::function3(ngl::Vec2 uv, float time)
{
#ifdef USE_CONE_F3

  float t_min = -10.0f;
  float t_max =  10.0f;
  float K = (time - t_min) / (t_max - t_min);

  float r1 = ( rad1 / resolution_x );
  float r2 = ( rad2 / resolution_x );

  float X_C = center1[0] + K * (center2[0] - center1[0]);
  float Y_C = center1[1] + K * (center2[1] - center1[1]);
  float R  = r1 + K * (r2 - r1);

  float ddd = R * R - std::pow( uv.m_x - X_C, 2.0f ) -
                      std::pow( uv.m_y - Y_C, 2.0f );

  float f3 = INTERSECT( INTERSECT( ddd, 10.0f + time ), ( 10.0f - time ) );

#endif //USE_CONE_F3

#ifdef USE_PYRAMID_F3

  float rec1_w = static_cast<float>( rec1.width )  / resolution_x;
  float rec1_h = static_cast<float>( rec1.height ) / resolution_y;

  float rec2_w = static_cast<float>( rec2.width )  / resolution_y;
  float rec2_h = static_cast<float>( rec2.height ) / resolution_x;

  float t_min = -10.0f;
  float t_max =  10.0f;
  float K = ( time - t_min ) / ( t_max - t_min );

  float rec_w = rec1_w + K * (rec2_w - rec1_w);
  float rec_h = rec1_h + K * (rec2_h - rec1_h);

  float X_C   = center_r1[0] + K * (center_r2[0] - center_r1[0]);
  float Y_C   = center_r1[1] + K * (center_r2[1] - center_r1[1]);

  float ddd = INTERSECT( rec_w - std::abs(uv.m_x - X_C),
                         rec_h - std::abs(uv.m_y - Y_C) );

  float f3 = INTERSECT( INTERSECT( ddd, 10.0f + time ), ( 10.0f - time ) );

#endif //USE_PYRAMID_F3

#ifdef USE_EGG_SHAPE_F3
  uv.m_y = 1.0f - uv.m_y;
  //ngl::Vec2 pos = ngl::Vec2(uv_x * 10.0f - 5.0f, uv_y * 10.0f - 3.0f);

  float R = ((rad1 + rad2) / 2.0f) * static_cast<float>(resolution_x);
  float a = R;

  float dx_c = (center1[0] - center2[0]);
  float dy_c = (center1[1] - center2[1]);
  float f    = std::sqrt(dx_c * dx_c + dy_c * dy_c) / 2.0f;
  float b    = std::sqrt(f*f + a*a);

  float dx = uv.m_x - dx_c / 2.0f;
  float dy = uv.m_y - dy_c / 2.0f;
  float tx = std::exp(0.2f * uv.m_x);
  float alpha = std::atan2(center2[1]-center1[1], center2[0]-center1[0]);

  float dist = (std::pow(dx*std::cos(alpha) - dy*std::sin(alpha), 2.0f) / a*a)+
               (std::pow(dx*std::sin(alpha) + dy*std::cos(alpha), 2.0f) / b*b) * tx - 1.0f;
  float f3 = INTERSECT( INTERSECT( dist, 10.0f + time ), ( 10.0f - time ) );
#endif

#ifndef USE_CONE_F3
#ifndef USE_PYRAMID_F3
#ifndef USE_EGG_SHAPE_F3
  float f3 = INTERSECT(time + 10.0f, 10.0f - time);
#endif
#endif
#endif // if not defined(USE_CONE_F3) || not defined(USE_PYRAMID_F3)

  return f3;
}


/*
 * functions below are providing color conversion rules between RGB/HSV color spaces;
 */
void shader::ColorShader::convert_from_RGB(float *r, float *g, float *b)
{
  switch (color_mode_name)
  {
    case ColorMode::RGB:
      break;
    case ColorMode::HSV:
      convert_RGB_to_HSV(r, g, b);
      break;
    case ColorMode::CIELab:
      convert_RGB_to_CIELab(r, g, b);
      break;
    default:
      break;
  }
}

void shader::ColorShader::convert_to_RGB(float *r, float *g, float *b)
{
  switch (color_mode_name)
  {
    case ColorMode::RGB:
      break;
    case ColorMode::HSV:
      convert_HSV_to_RGB(r, g, b);
      break;
    case ColorMode::CIELab:
      convert_CIELab_to_RGB(r, g, b);
      break;
    default:
      break;
  }
}

void shader::ColorShader::convert_RGB_to_HSV(float *r, float *g, float *b)
{
  float min, max, delta;
  float h, s, v;

  min = min3( *r, *g, *b );
  max = max3( *r, *g, *b );

  delta = max - min;
  v     = max;

  if( max != 0.0f )
    s = delta / max;
  else
  {
    h = 0.0f;
    s = 0.0f;
    v = 0.0f;
  }

  if( *r == max && *r != 0.0f )
    h = ( *g - *b ) / delta;        // between yellow & magenta

  else if( *g == max && *g != 0.0f )
    h = 2.0f + ( *b - *r ) / delta;    // between cyan & yellow

  else if( *b == max && *b != 0.0f )
    h = 4.0f + ( *r - *g ) / delta;    // between magenta & cyan

  h *= 60.0f;    // degrees
  if( h < 0.0f )
    h += 360.0f;

  if( h > 360.0f )
    h -= 360.0f;

  h /= 360.0f;

  *r = h;
  *g = s;
  *b = v;
}

void shader::ColorShader::convert_HSV_to_RGB(float *r, float *g, float *b)
{
  int i;
  float f, p, q, t;
  float r1, g1, b1;

  if( *g == 0.0f )
  {
    r1 = g1 = b1 = *b;
    *r = *g = *b = r1;
  }
  else
  {
    *r *= 360.0f;
    *r /= 60.0f;  // sector 0 to 5

    i = static_cast<int>(std::floor( *r ));
    f = *r - i;              // factorial part of h
    p = *b * ( 1.0f - *b );
    q = *b * ( 1.0f - *b * f );
    t = *b * ( 1.0f - *b * ( 1.0f - f ) );

    switch( i )
    {
      case 0:
        r1 = *b;
        g1 = t;
        b1 = p;
        break;
      case 1:
        r1 = q;
        g1 = *b;
        b1 = p;
        break;
      case 2:
        r1 = p;
        g1 = *b;
        b1 = t;
        break;
      case 3:
        r1 = p;
        g1 = q;
        b1 = *b;
        break;
      case 4:
        r1 = t;
        g1 = p;
        b1 = *b;
        break;
      default:
        r1 = *b;
        g1 = p;
        b1 = q;
        break;
    }

    *r = r1;
    *g = g1;
    *b = b1;
  }
}

void shader::ColorShader::convert_RGB_to_CIELab(float *r, float *g, float *b)
{
    //1 step is to convert RGB to XYZ color space
    float r1, g1, b1;
    if(*r > 0.04045f) r1 = std::pow(((*r+0.055f) / 1.055f), 2.4f);
    else                    r1 = *r / 12.92f;

    if(*g > 0.04045f) g1 = std::pow(((*g+0.055f) / 1.055f), 2.4f);
    else                    g1 = *g / 12.92f;

    if(*b > 0.04045f) b1 = std::pow(((*b+0.055f) / 1.055f), 2.4f);
    else                    b1 = *b / 12.92f;

    r1 = r1 * 100.0f;
    g1 = g1 * 100.0f;
    b1 = b1 * 100.0f;

    float X = r1 * 0.4124f + g1 * 0.3576f + b1 * 0.1805f;
    float Y = r1 * 0.2126f + g1 * 0.7152f + b1 * 0.0722f;
    float Z = r1 * 0.0193f + g1 * 0.1192f + b1 * 0.9505f;

    //2 step is to convert XYZ to CIELab color space
    //CIELab is dependong on standart observer parameters for the standart type of light (in our case D65)
    //chosen observer: 2 degree, Illuminant = D65;
    float var_X = X / 95.047f;
    float var_Y = Y / 100.0f;
    float var_Z = Z / 108.883f;

    if ( var_X > 0.008856f ) var_X = std::pow(var_X, 1.0f/3.0f );
    else                     var_X = ( 7.787f * var_X ) + ( 16.0f / 116.0f );

    if ( var_Y > 0.008856f ) var_Y = std::pow(var_Y, 1.0f/3.0f );
    else                     var_Y = ( 7.787f * var_Y ) + ( 16.0f / 116.0f );

    if ( var_Z > 0.008856f ) var_Z = std::pow(var_Z, 1.0f/3.0f );
    else                     var_Z = ( 7.787f * var_Z ) + ( 16.0f / 116.0f );

    *r = 116.0f * var_Y - 16.0f;
    *g = 500.0f * (var_X - var_Y);
    *b = 200.0f * (var_Y - var_Z);
}

void shader::ColorShader::convert_CIELab_to_RGB(float *r, float *g, float *b)
{
  // 1 step: converting CIELab to XYZ color space
  float var_Y = (*r + 16.0f) / 116.0f;
  float var_X = *g / 500.0f + var_Y;
  float var_Z = var_Y - *b / 200.0f;

  if(pow(var_Y, 3.0f) > 0.008856f) var_Y = std::pow(var_Y, 3.0f);
  else                             var_Y = (var_Y- 16.0f/116.0f) / 7.787f;

  if(pow(var_X, 3.0f) > 0.008856f) var_X = std::pow(var_X, 3.0f);
  else                             var_X = (var_X- 16.0f/116.0f) / 7.787f;

  if(pow(var_Z, 3.0f) > 0.008856f) var_Z = std::pow(var_Z, 3.0f);
  else                             var_Z = (var_Z- 16.0f/116.0f) / 7.787f;

  //standart observer: 2 degree + illuminant = D65
  float X = var_X * 95.047f;
  float Y = var_Y * 100.0f;
  float Z = var_Z * 108.883f;

  // 2 step: converting XYZ to RGB color space
  X = X / 100.0f;
  Y = Y / 100.0f;
  Z = Z / 100.0f;

  float var_R = X * 3.2406f    + Y * (-1.5372f) + Z * (-0.4986f);
  float var_G = X * (-0.9686f) + Y * 1.8758f    + Z * 0.0415f;
  float var_B = X * 0.0557f    + Y * (-0.2040f) + Z * 1.0570f;

  if ( var_R > 0.0031308f ) var_R = 1.055f * std::pow( var_R, 1.0f / 2.4f ) - 0.055f;
  else                      var_R = 12.92f * var_R;
  if ( var_G > 0.0031308f ) var_G = 1.055f * std::pow( var_G, 1.0f / 2.4f ) - 0.055f;
  else                      var_G = 12.92f * var_G;
  if ( var_B > 0.0031308f ) var_B = 1.055f * std::pow( var_B, 1.0f / 2.4f ) - 0.055f;
  else                      var_B = 12.92f * var_B;

  *r = var_R;
  *g = var_G;
  *b = var_B;
}

#ifdef USE_AFFINE_TRANSFORMATIONS
cv::Mat shader::ColorShader::update_affine_rotation_m(cv::Point2f sh_center, double step)
{
    angle +=step;
    rot_affine_m = (cv::Mat_<double>(2 ,3) << 0, 0, 0, 0, 0, 0);
    rot_affine_m.at<double>(0, 0) = std::cos(angle * M_PI / 180.0);
    rot_affine_m.at<double>(0, 1) = std::sin(angle * M_PI / 180.0);
    rot_affine_m.at<double>(0, 2) = (1.0 - std::cos(angle * M_PI / 180.0)) * sh_center.x - std::sin(angle * M_PI / 180.0) * sh_center.y;

    rot_affine_m.at<double>(1, 0) = -std::sin(angle * M_PI / 180.0);
    rot_affine_m.at<double>(1, 1) =  std::cos(angle * M_PI / 180.0);
    rot_affine_m.at<double>(1, 2) =  std::sin(angle * M_PI / 180.0) * sh_center.x + (1.0 - std::cos(angle * M_PI / 180.0)) * sh_center.y;

    return rot_affine_m;
}
#endif
