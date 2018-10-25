#ifdef USE_OPENMP
  #include <openmp/include/omp.h>
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

shader::ColorShader::ColorShader(int length, cv::Mat img1, cv::Mat img2,
                                 ColorMode color_mode, DrawMode draw_mode)
{
    length_param     = length;
    draw_mode_name   = draw_mode;
    color_mode_name  = color_mode;

    img1.copyTo(image1);
    img2.copyTo(image2);
    im1_buf = static_cast<unsigned char*>(image1.data);
    im2_buf = static_cast<unsigned char*>(image2.data);
    resolution_x = image1.cols;
    resolution_y = image1.rows;

    DF = nullptr;
    set_ai_coeffs();

    auto start = std::chrono::high_resolution_clock::now();

    DF = std::make_shared<distance_field::DistanceField>(image1, image2);

    std::vector<double> SDDT11 = DF.get()->get_SDDT1();
    double *SDDT1 = SDDT11.data();
    F1.resize(resolution_x*resolution_y);
    float *FF1 = F1.data();

    std::vector<double> SDDT22 = DF.get()->get_SDDT2();
    double *SDDT2 = SDDT22.data();
    F2.resize(resolution_x*resolution_y);
    float *FF2 = F2.data();

    double u, v, u_opp, v_opp;

#ifdef USE_OPENMP
#pragma omp parallel shared(SDDT1, SDDT2, FF1, FF2, resolution_x, resolution_y) private(u,v,u_opp,v_opp)
{
#pragma omp for simd schedule(static,5)
#endif
    for(int y= 0; y < resolution_y; ++y)
    {
        for(int x = 0; x < resolution_x; ++x)
        {
            u = static_cast<double>(x / (resolution_x+1));
            v = static_cast<double>(y / (resolution_y+1));

            u_opp = 1.0 - u;
            v_opp = 1.0 - v;

            FF1[x+y*resolution_x] = ( SDDT1[x+    y* (resolution_x+1)] * u_opp + SDDT1[(x+1)+    y* (resolution_x+1)] * u ) * v_opp +
                                    ( SDDT1[x+(y+1)* (resolution_x+1)] * u_opp + SDDT1[(x+1)+(y+1)* (resolution_x+1)] * u ) * v;
        }
    }

#ifdef USE_OPENMP
#pragma omp for simd schedule(static,5)
#endif

    for(int y= 0; y < resolution_y; ++y)
    {
        for(int x = 0; x < resolution_x; ++x)
        {
            u = static_cast<double>(x / (resolution_x+1));
            v = static_cast<double>(y / (resolution_y+1));

            u_opp = 1.0 - u;
            v_opp = 1.0 - v;

            FF2[x+y*resolution_x] = ( SDDT2[x+    y*(resolution_x+1)] * u_opp + SDDT2[(x+1)+    y*(resolution_x+1)] * u ) * v_opp +
                                    ( SDDT2[x+(y+1)*(resolution_x+1)] * u_opp + SDDT2[(x+1)+(y+1)*(resolution_x+1)] * u ) * v;
        }
    }
#ifdef USE_OPENMP
}
#endif

#ifndef USE_AFFINE_TRANSFORMATIONS
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Preproc. time (around distance_fields): " << dur.count() << " seconds" << std::endl;
#endif
#if defined(USE_AFFINE_TRANSFORMATIONS) || defined(USE_CONE_F3) || defined(USE_PYRAMID_F3) || defined(USE_EGG_SHAPE_F3)

#if defined(USE_CONE_F3) || defined(USE_EGG_SHAPE_F3)
    stb_tricks::StbEnhancements stb(image1, image2);
    stb.get_max_circle(&rad1, &rad2, &center1, &center2);
    r1 = ( rad1 / resolution_x );
    r2 = ( rad2 / resolution_x );
#endif //USE_CONE_F3

#ifdef USE_PYRAMID_F3
    stb_tricks::StbEnhancements stb(image1, image2);
    rec1 = stb.get_max_size_rectangle(stb.get_rectangle_box_1());
    rec2 = stb.get_max_size_rectangle(stb.get_rectangle_box_2());
    center_r1 = stb.get_rectangle_center_coords_1();
    center_r2 = stb.get_rectangle_center_coords_2();
    rec1_w = static_cast<float>( rec1.width )  / resolution_x;
    rec1_h = static_cast<float>( rec1.height ) / resolution_y;
    rec2_w = static_cast<float>( rec2.width )  / resolution_y;
    rec2_h = static_cast<float>( rec2.height ) / resolution_x;
#endif //USE_PYRAMID_F3

#ifdef USE_AFFINE_TRANSFORMATIONS
    stb_tricks::StbEnhancements stb(image1, image2, true);
    image2    = stb.get_affine_transformed_image(&affine_m);
    center_r1 = stb.get_rectangle_center_coords_1();
    center_r2 = stb.get_rectangle_center_coords_2();

    im2_buf = static_cast<unsigned char*>(image2.data);
    F2_m.resize(F2.size());
    float *F22_m = F2_m.data();
    float u_x, u_y;
    float res = 0.0f;

    for(int y = 0; y < resolution_y; ++y)
    {
        for(int x = 0; x < resolution_x; ++x)
        {
            u_x = static_cast<float>(x)/static_cast<float>(resolution_x);
            u_y = static_cast<float>(y)/static_cast<float>(resolution_y);

            DF.get()->inverse_mapping_result(&res, F2, u_x, u_y, affine_m);
            F22_m[x+y*resolution_x] = res;
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

    cx1 = cen1[0] ;
    cy1 = cen1[1] ;
    cx2 = cen2[0] ;
    cy2 = cen2[1] ;
    r1_uv = r1 / resolution_x;
    r2_uv = r2 / resolution_x;

#ifndef USE_MANUAL_ai
    a1 = calc_a1_coeff();
    a2 = a1;
#endif

#ifdef USE_CONE_F3
    a3 = std::min(r1_uv, r2_uv)*0.03126f;
#elif defined (USE_PYRAMID_F3)
    a3 = std::min(r1_uv, r2_uv)*0.13336f;
#else
    a3 = max_a3;
#endif

#ifndef USE_MANUAL_ai
    std::vector<float> a0_interval = calc_a0_interval();
    a0 = a0_interval[1];

    std::cout << "a1 = a2 = " << a1 << std::endl;
    std::cout << "     a3 = " << a3 << std::endl;
    std::cout << "     a0 = " << a0 << std::endl;

    //std::cout << "Enter a_0 suitable for you: ";
    //std::cin >> a0; std::cout << std::endl;
#endif

}

bool shader::ColorShader::check_2circle_crosssection()
{
    float coef1 = cx1*cx1 + cy1*cy1 - r1_uv*r1_uv;
    float coef2 = cx2*cx2 + cy2*cy2 - r2_uv*r2_uv;

    float c1 = 4.0f*( (cx1 - cx2)*(cx1 - cx2) + (cy1 - cy2)*(cy1 - cy2) );
    float c2 = (coef1 - coef2)*(coef1 - coef2) + cy1*cy1*(4.0f*coef2 - 8.0f*coef1) +
               4.0f*cy1*cy2*(3.0f*coef1 - coef2) - 4.0f*coef1*cy2*cy2;
    float c3 = 4.0f*coef1*(cx2 - cx1) + 4.0f*coef2*(cx1 - cx2) - 8.0f*cx2*(cy1*cy1 - cy1*cy2)
                                                                - 8.0f*cx1*(cy2*cy2 - cy1*cy2);

    float diskr = -4.0f*c1*c2 + c3*c3;

    if(diskr < 0.0f && std::sqrt( (cx2-cx1)*(cx2-cx1) + (cy2-cy1)*(cy2-cy1)) < std::min(r1_uv, r2_uv)/2.0f )
        return false;
    else if (diskr < 1.0f/100000.0f)
        return false;
    else if (diskr == 0.0f)
        return false;
    else
        return true;
}

float shader::ColorShader::calc_a1_coeff()
{
    //****************************************************************************************
    //STEP I: find middle points of the line connected centers of two circles for estimating a1
    float a1_fin;
    float dist = std::sqrt( (cx2-cx1)*(cx2-cx1) + (cy2-cy1)*(cy2-cy1));
    if( dist < 0.01f)
    {
        a1_fin = 1.0;
    }
    else {
        //calculating point K1
        //coeffs for K1x
        float k = 1.0f / ( (cx1-cx2)*(cx1-cx2) + (cy1-cy2) * (cy1-cy2) );
        float k1_1 = std::pow(cx1, 3.0f) - 2*cx1*cx1*cx2 + cx1*cx2*cx2 +
                             cx1*cy1*cy1 - 2*cx1*cy1*cy2 + cx1*cy2*cy2;

        float k1_2 = std::pow(cx1, 4.0f) - 4*std::pow(cx1, 3.0f)*cx2 +
                     6*cx1*cx1*cx2*cx2   - 4*cx1*std::pow(cx2, 3.0f) +
                     std::pow(cx2, 4.0f) + cx1*cx1*cy1*cy1 -
                     2*cx1*cx2*cy1*cy1   + cx2*cx2*cy1*cy1 -
                     2*cx1*cx1*cy1*cy2   + 4*cx1*cx2*cy1*cy2 -
                     2*cx2*cx2*cy1*cy2   + cx1*cx1*cy2*cy2 -
                     2*cx1*cx2*cy2*cy2   + cx2*cx2*cy2*cy2;

        //coeffs for K1y
        float k1_3 = std::pow(cx1, 3.0f)*cy1 - 2*cx1*cx1*cx2*cy1 + cx1*cx2*cx2*cy1 +
                     cx1*std::pow(cy1, 3.0f) - std::pow(cx1, 3.0f)*cy2 +
                     2*cx1*cx1*cx2*cy2       - cx1*cx2*cx2*cy2 - 3*cx1*cy1*cy1*cy2 +
                     3*cx1*cy1*cy2*cy2       - cx1*std::pow(cy2, 3.0f);

        float K1x_a = k * (k1_1 - r1_uv * std::sqrt(k1_2));
        float K1y_a = ( 1.0f/( cx1 - cx2 ) ) * ( -cx2*cy1 + cx1*cy2 + k * ( k1_3 - (cy1-cy2) * r1_uv * std::sqrt(k1_2) ));

        float K1x_b = k * (k1_1 + r1_uv * std::sqrt(k1_2));
        float K1y_b = ( 1.0f/( cx1 - cx2 ) ) * ( -cx2*cy1 + cx1*cy2 + k * ( k1_3 + (cy1-cy2) * r1_uv * std::sqrt(k1_2) ));

        //****************************************************************************************
        //calculating point K2
        //coeffs for K2x1
        float k2_1 = cx1*cx1*cx2 - 2*cx1*cx2*cx2 + std::pow(cx2, 3.0f) +
                     cx2*cy1*cy1 - 2*cx2*cy1*cy2 + cx2*cy2*cy2;
        float k2_2 = k1_2;

        //coeffs for K2y
        float k2_3 = cx1*cx1*cx2*cy1         - 2*cx1*cx2*cx2*cy1 +
                     std::pow(cx2, 3.0f)*cy1 + cx2*std::pow(cy1, 3.0f) -
                     cx1*cx1*cx2*cy2         + 2*cx1*cx2*cx2*cy2 - std::pow(cx2, 3.0f)*cy2 -
                     3*cx2*cy1*cy1*cy2       + 3*cx2*cy1*cy2*cy2 -
                     cx2*std::pow(cy2, 3.0f);

        float K2x_a = k * ( k2_1 - r2_uv * std::sqrt(k2_2) );
        float K2y_a = ( 1.0f/( cx1 - cx2 ) ) * ( -cx2*cy1 + cx1*cy2 + k * ( k2_3 - (cy1-cy2) * r2_uv * std::sqrt(k2_2) ));

        float K2x_b = k * ( k2_1 + r2 * std::sqrt(k2_2) );
        float K2y_b = ( 1.0f/( cx1 - cx2 ) ) * ( -cx2*cy1 + cx1*cy2 + k * ( k2_3 + (cy1-cy2) * r2_uv * std::sqrt(k2_2) ));

        //calculating norm of the K1K2 vector
        float norm1 = (K2x_a - K1x_a)*(K2x_a - K1x_a) + (K2y_a - K1y_a)*(K2y_a - K1y_a);
        float norm2 = (K2x_b - K1x_b)*(K2x_b - K1x_b) + (K2y_b - K1y_b)*(K2y_b - K1y_b);

        float norm, K2x, K2y, K1x, K1y;
        if(norm1 < norm2)
        {
            norm = norm1;
            K1x = K1x_a; K1y = K1y_a;
            K2x = K2x_a; K2y = K2y_a;
        }
        else
        {
            norm = norm2;
            K1x = K1x_b; K1y = K1y_b;
            K2x = K2x_b; K2y = K2y_b;
        }

        //calculating f1 and f2 in these points
        float f1_k = r1_uv - std::sqrt((K2x - cx1)*(K2x - cx1)) - std::sqrt((K2y - cy1)*(K2y - cy1));
        float f2_k = r2_uv - std::sqrt((K1x - cx2)*(K1x - cx2)) - std::sqrt((K1y - cy2)*(K1y - cy2));

        //float f1_k = (K2x - cx1)*(K2x - cx1) + (K2y - cy1)*(K2y - cy1) - r1_uv*r1_uv;
        //float f2_k = (K1x - cx2)*(K1x - cx2) + (K1y - cy2)*(K1y - cy2) - r2_uv*r2_uv;

        a1_fin = std::sqrt( (f1_k*f1_k + f2_k*f2_k) / norm );
    }
    return a1_fin;
}

std::vector<float> shader::ColorShader::calc_a0_interval()
{
    float cx_min = std::min(cx1, cx2);
    if( cx_min == cx1 )
    {
        x_min = cx_min - r1_uv;
        x_max = cx2 + r2_uv;
    }
    else
    {
        x_min = cx_min - r2_uv;
        x_max = cx1 + r1_uv;
    }

    float cy_min = std::min(cy1, cy2);
    if(cy_min == cy1 )
    {
        y_min = cy_min - r1_uv;
        y_max = cy2 + r2_uv;
    }
    else
    {
        y_min = cy_min - r2_uv;
        y_max = cy1 + r1_uv;
    }

    if(x_min < 0.0f) x_min = 0.0f;
    if(x_max > 1.0f) x_max = 1.0f;
    if(y_min < 0.0f) y_min = 0.0f;
    if(y_max > 1.0f) y_max = 1.0f;

    //intervals for sqrt[x*x-2xx1+x1*x1] and sqrt[y*y-2yx1+y1*y1]; sqrt[x*x-2xx2+x2*x2] and sqrt[y*y-2yx2+y2*y2]
    float x1_min = x_min*x_max - 2.0f*cx1*x_max + cx1*cx1;
    float y1_min = y_min*y_max - 2.0f*cy1*y_max + cy1*cy1;
    float x2_min = x_min*x_max - 2.0f*cx2*x_max + cx2*cx2;
    float y2_min = y_min*y_max - 2.0f*cy2*y_max + cy2*cy2;

    if(x1_min < 0.0f) x1_min = 0.0f;
    if(y1_min < 0.0f) y1_min = 0.0f;
    if(x2_min < 0.0f) x2_min = 0.0f;
    if(y2_min < 0.0f) y2_min = 0.0f;

    float x1_max = x_max*x_max - 2.0f*cx1*x_min + cx1*cx1;
    float y1_max = y_max*y_max - 2.0f*cy1*y_min + cy1*cy1;
    float x2_max = x_max*x_max - 2.0f*cx2*x_min + cx2*cx2;
    float y2_max = y_max*y_max - 2.0f*cy2*y_min + cy2*cy2;

    //intervals for function f1 and f2 represented as circle equations
    fun1 = { std::sqrt(x1_min) + std::sqrt(y1_min) - r1_uv,
             std::sqrt(x1_max) + std::sqrt(y1_max) - r1_uv };
    fun2 = { std::sqrt(x2_min) + std::sqrt(y2_min) - r2_uv,
             std::sqrt(x2_max) + std::sqrt(y2_max) - r2_uv };

    //f1*f1 and f2*f2 check if the min value is positive, otherwise it is 0.0
    float fun11_min = fun1[0]*fun1[1];
    float fun22_min = fun2[0]*fun2[1];

    if( fun11_min < 0.0f ) fun11_min = 0.0f;
    if( fun22_min < 0.0f ) fun22_min = 0.0f;

    fun11 = { fun11_min, fun1[1]*fun1[1] };
    fun22 = { fun22_min, fun2[1]*fun2[1] };

    //checking whether sum under the square root is positive: sqrt(f1*f1 + f2*f2)

    f11f22_sum    = { fun11[0] + fun22[0], fun11[1] + fun22[1] };
    SqrF12F22_sum = { std::sqrt(f11f22_sum[0]), std::sqrt(f11f22_sum[1]) };

    //calculating final estimation interval for a0
    f1f2_sum      = { fun1[0] + fun2[0] , fun1[1] + fun2[1] };


    std::vector<float> a0_bot, a0_top;

    if(check_2circle_crosssection())
    {
        float dy = ( cy2 - cy1 );
        float dx = ( cx2 - cx1 );
        float dr = ( r2_uv - r1_uv );
        float d  = std::sqrt( dx*dx + dy*dy );

        float X = dx / d;
        float Y = dy / d;
        float R = dr / d;

        //for top tangential line
        A1 = R*X - Y*std::sqrt( 1.0f - R*R );
        B1 = R*Y + X*std::sqrt( 1.0f - R*R );
        C1 = r1_uv - ( A1*cx1 + B1*cy1 );

        //for bottom tangential line
        A2 = R*X + Y*std::sqrt( 1.0f - R*R );
        B2 = R*Y - X*std::sqrt( 1.0f - R*R );
        C2 = r1_uv - ( A2*cx1 + B2*cy1 );

        std::vector<float> AA = { f1f2_sum[0] + SqrF12F22_sum[0], f1f2_sum[1] + SqrF12F22_sum[1] };
        std::vector<float> BB = { 1.0f + (1.0f/(a1*a1))*f11f22_sum[0], 1.0f + (1.0f/(a1*a1))*f11f22_sum[1]};
        std::vector<float> CC1 = { A1*x_min + B1*y_min + C1, A1*x_max + B1*y_max + C1 };
        std::vector<float> CC2 = { A2*x_min + B2*y_min + C2, A2*x_max + B2*y_max + C2 };

        std::vector<float> DD1 = { CC1[0] - AA[1], CC1[1] - AA[0] };
        std::vector<float> DD2 = { CC2[0] - AA[1], CC2[1] - AA[0] };

        //found bottom interval for a0
        a0_bot = { min4(DD2[0]*BB[0], DD2[0]*BB[1], DD2[1]*BB[0], DD2[1]*BB[1]),
                   max4(DD2[0]*BB[0], DD2[0]*BB[1], DD2[1]*BB[0], DD2[1]*BB[1]) };

        a0_top = { min4(DD1[0]*BB[0], DD1[0]*BB[1], DD1[1]*BB[0], DD1[1]*BB[1]),
                   max4(DD1[0]*BB[0], DD1[0]*BB[1], DD1[1]*BB[0], DD1[1]*BB[1]) };

        float a0_min, a0_max;
        if(a0_bot[0] < 0.0f && a0_top[0] < 0.0f && a0_bot[1] > 0.0f)
            a0_min = a0_bot[1];
        else
            a0_min = a0_bot[0];

        a0_max = a0_top[1];

        a0_int = { a0_min, a0_max };

        std::cout << "bottom interval [1] for a0 [ " << a0_bot[0] << " , " << a0_bot[1] << " ]" << std::endl;
        std::cout << "top    interval [2] for a0 [ " << a0_top[0] << " , " << a0_top[1] << " ]" << std::endl;
        std::cout << "final  interval for a0 [ "     << a0_min    << " , " << a0_max    << " ]" << std::endl;
    }
    else
    {
        float cx, cy, r_uv;
        if(std::max(r1_uv, r2_uv) == r1_uv)
        {
            cx = cx1;
            cy = cy1;
            r_uv = r1_uv;
        }
        else
        {
            cx = cx2;
            cy = cy2;
            r_uv = r2_uv;
        }

        std::vector<float> AA = { f1f2_sum[0] + SqrF12F22_sum[0], f1f2_sum[1] + SqrF12F22_sum[1] };
        std::vector<float> BB = { 1.0f + (1.0f/(a1*a1))*f11f22_sum[0], 1.0f + (1.0f/(a1*a1))*f11f22_sum[1]};
        std::vector<float> CC = { (x_min - cx)*(x_max - cx) + (y_min - cy)*(y_max - cy) - r_uv*r_uv,
                                  (x_max - cx)*(x_max - cx) + (y_max - cy)*(y_max - cy) - r_uv*r_uv };
        std::vector<float> DD = { CC[0] - AA[1], CC[1] - AA[0] };

        a0_int = { min4(DD[0]*BB[0], DD[1]*BB[0], DD[0]*BB[1], DD[1]*BB[1]),
                   max4(DD[0]*BB[0], DD[1]*BB[0], DD[0]*BB[1], DD[1]*BB[1]) };

        std::cout << "final  interval for a0 [ " << a0_int[0]    << " , " << a0_int[1]    << " ]" << std::endl;
    }


    return a0_int;
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
    size_t step = img->step;

    b = img_buf[x+  y*step];
    g = img_buf[x+1+y*step];
    r = img_buf[x+2+y*step];
    a = img_buf[x+3+y*step];

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
  fun1 = F1[static_cast<int>(uv.m_x*resolution_x) + static_cast<int>(uv.m_y*resolution_y*resolution_x)];
#ifdef USE_AFFINE_TRANSFORMATIONS
  fun2 = F2_m[static_cast<int>(uv.m_x*resolution_x) + static_cast<int>(uv.m_y*resolution_y*resolution_x)];
#else
  fun2 = F2[static_cast<int>(uv.m_x*resolution_x) + static_cast<int>(uv.m_y*resolution_y*resolution_x)];
#endif

  f1 = smooth_cylynder1(time, fun1);
  f2 = smooth_cylynder2(time, fun2);
#else
  fun1 = F1[static_cast<int>(uv.m_x*resolution_x) + static_cast<int>(uv.m_y*resolution_y*resolution_x)];
  fun2 = F2[static_cast<int>(uv.m_x*resolution_x) + static_cast<int>(uv.m_y*resolution_y*resolution_x)];

  f1 = INTERSECT(fun1, -time);
  f2 = INTERSECT(fun2, (time - 1.0f));
#endif //USE_SMOOTHED_CYLINDERS

  float f3 = function3(uv, time);

  float r1 = (f1/a1)*(f1/a1) + (f2/a2)*(f2/a2);
  float r2 = 0.0f;

  if( f3 > 0.0f )
  {
    r2 = (f3/a3) * (f3/a3);
  }

  float rr = 0.0f;
  if( r1 > 0.0f )
  {
    rr = r1 / (r1 + r2);
  }

  float d = 0.0f;
  if( rr < 1.0f )
  {
    d = a0 * (1.0f - rr)*(1.0f - rr)*(1.0f - rr) / (1.0f + rr);
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

float shader::ColorShader::smooth_cylynder1(float time, float fun)
{
#if defined (USE_PYRAMID_F3) || defined(USE_CONE_F3)
#ifdef USE_PYRAMID_F3
    b1_0 = f3_pyr_1;
#else
    b1_0 = f3_cone_1;
#endif
#endif

    float f1 = fun;
    float f2 = -time;
    float f3 =  time + 5.0f;

    float r1 = (f1/b1_1)*(f1/b1_1) + (f2/b1_2)*(f2/b1_2);
    float r2 = 0.0f;

    if( f3 > 0.0f )
    {
      r2 = (f3/b1_3) * (f3/b1_3);
    }

    float rr = 0.0f;
    if( r1 > 0.0f )
    {
      rr = r1 / (r1 + r2);
    }

    float d = 0.0f;
    if( rr < 1.0f )
    {
      d = b1_0 * (1.0f - rr)*(1.0f - rr)*(1.0f - rr) / (1.0f + rr);
    }

    float blending_result = INTERSECT(f1, f2) + d;
    return blending_result;
}

float shader::ColorShader::smooth_cylynder2(float time, float fun)
{
#if defined (USE_PYRAMID_F3) || defined(USE_CONE_F3)
#ifdef USE_PYRAMID_F3
    float b2_0 = f3_pyr_2;
#else
    float b2_0 = f3_cone_2;
#endif
#endif

    float f1 = fun;
    float f2 = time - 1.0f;
    float f3 = 5.0f - time;

    float r1 = (f1/b2_1)*(f1/b2_1) + (f2/b2_2)*(f2/b2_2);
    float r2 = 0.0f;

    if( f3 > 0.0f )
    {
      r2 = (f3/b2_3) * (f3/b2_3);
    }

    float rr = 0.0f;
    if( r1 > 0.0f )
    {
      rr = r1 / (r1 + r2);
    }

    float d = 0.0f;
    if( rr < 1.0f )
    {
      d = b2_0 * (1.0f - rr)*(1.0f - rr)*(1.0f - rr) / (1.0f + rr);
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
  int res_x = resolution_x;
  int res_y = resolution_y;

  unsigned char *buff_im1 = im1_buf;
  unsigned char *buff_im2 = im2_buf;

  if(result1 < 0.0f)
   {
       float total_w1 = 0.0f;
       float r1 = 0;
       float g1 = 0;
       float b1 = 0;
       int ch = image1.channels();
       size_t step = image1.step;
       float *data = F1.data();
       float b,g,r,x1, y1, dist;

#ifdef USE_OPENMP
#ifdef USE_OPENMP_WITH_CUDA_CLANG
#pragma omp target map(to: data[0: F1.size()], buff_im1[0:res_x*res_y*ch]) \
                        map(to: u_x, u_y, ch, step, res_x, res_y, b,g,r,x1,y1,dist)
{
#pragma omp teams distribute parallel for simd reduction(+:total_w1,r1,g1,b1) schedule(static,1)
#else
#pragma omp parallel shared(b1,r1,g1,total_w1,data,buff_im1,ch,step,res_x, res_y) private(b,g,r,x1,y1,dist)
{
#pragma omp for simd reduction(+:total_w1, r1, g1, b1) schedule(static, 5)
#endif
#endif
      for(int i1 = 0; i1 < res_y; ++i1)
      {
         for(int j1 = 0; j1 < res_x; ++j1)
         {
            if(data[j1+i1*res_x] >= 0.0f)
            {
                b = buff_im1[j1*ch+  i1*step] / 255.0f;
                g = buff_im1[j1*ch+1+i1*step] / 255.0f;
                r = buff_im1[j1*ch+2+i1*step] / 255.0f;
                convert_from_RGB(&r, &g, &b);

                x1 = static_cast<float>(j1) / static_cast<float>(res_x);
                y1 = static_cast<float>(i1) / static_cast<float>(res_y);
                dist = ( (u_x - x1) * (u_x - x1) + (u_y - y1) * (u_y - y1) ) *
                       ( (u_x - x1) * (u_x - x1) + (u_y - y1) * (u_y - y1) );

                b1 += b / dist;
                g1 += g / dist;
                r1 += r / dist;
                total_w1 += 1.0f / dist;
            }
         }
      }
#ifdef USE_OPENMP
}
#endif

      *col1 = ngl::Vec4(r1 / total_w1, g1 / total_w1, b1 / total_w1, 1.0f);
 }

  if(result2 < 0.0f)
  {
      float total_w2 = 0.0f;
      float r2 = 0;
      float g2 = 0;
      float b2 = 0;
      int ch = image1.channels();
      size_t step = image1.step;
#ifdef USE_AFFINE_TRANSFORMATIONS
      float *data = F2_m.data();
#else
      float *data = F2.data();
#endif
      float b,g,r,x2, y2, dist;

#ifdef USE_OPENMP
#ifdef USE_OPENMP_WITH_CUDA_CLANG
#pragma omp target map(to: buff_im2[:res_x*res_y*ch], data[: F2.size()])\
                        map(to: res_x, res_y, u_x, u_y, ch, step,b,g,r,x2,y2,dist)
{
#pragma omp teams distribute parallel for simd reduction(+:total_w2,r2,g2,b2) schedule(static,1)
#else

#pragma omp parallel shared(b2,r2,g2,total_w2,data,buff_im2,ch,step,res_x, res_y) private(b,g,r,x2,y2,dist)
{
#pragma omp for simd reduction(+:total_w2, r2, g2, b2) schedule(static, 5)
#endif
#endif
     for(int i2 = 0; i2 < res_y; ++i2)
     {
        for(int j2 = 0; j2 < res_x; ++j2)
        {
           if(data[j2+i2*res_x] >= 0.0f)
           {
              b = buff_im2[j2*ch+  i2*step] / 255.0f;
              g = buff_im2[j2*ch+1+i2*step] / 255.0f;
              r = buff_im2[j2*ch+2+i2*step] / 255.0f;
              convert_from_RGB(&r, &g, &b);

              x2 = static_cast<float>(j2) / static_cast<float>(res_x);
              y2 = static_cast<float>(i2) / static_cast<float>(res_y);
              dist = ( (u_x - x2) * (u_x - x2) + (u_y - y2) * (u_y - y2) ) *
                     ( (u_x - x2) * (u_x - x2) + (u_y - y2) * (u_y - y2) );

               b2 += b / dist;
               g2 += g / dist;
               r2 += r / dist;
               total_w2 += 1.0f / dist;
           }
        }
     }
#ifdef USE_OPENMP
}
#endif

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

  float *data1 = F1.data();
  float smallest_dist_squared1 = 10000.0f;

#ifdef USE_OPENMP
#pragma omp target teams distribute parallel for collapse(2) map(from: cl1_uv_x, cl1_uv_y) \
            map(to: u_x, u_y, smallest_dist_squared1) map(to: data1[:F1.size()])
#endif
  for(int i1 = 0; i1 < resolution_y; ++i1)
  {
      for(int j1 = 0; j1 < resolution_x; ++j1)
      {
         float x1 = static_cast<float>(j1)/static_cast<float>(resolution_x);
         float y1 = static_cast<float>(i1)/static_cast<float>(resolution_y);
         float dist_squared = (u_x - x1)*(u_x - x1) + (u_y - y1)*(u_y - y1);

         if(data1[j1+i1*resolution_x] >= 0.0f && dist_squared < smallest_dist_squared1)
         {
            smallest_dist_squared1 = dist_squared;
            cl1_uv_x = x1;
            cl1_uv_y = y1;
         }
      }
  }

  float cl2_uv_x, cl2_uv_y;
#ifdef USE_AFFINE_TRANSFORMATIONS
      float *data2 = F2_m.data();
#else
      float *data2 = F2.data();
#endif
      float smallest_dist_squared2 = 10000.0f;

#ifdef USE_OPENMP
#pragma omp target teams distribute parallel for collapse(2) map(from: cl2_uv_x, cl2_uv_y) \
            map(to: u_x, u_y,smallest_dist_squared1) map(to: data2[:F2.size()])
#endif
  for(int i2 = 0; i2 < resolution_y; ++i2)
  {
      for(int j2 = 0; j2 < resolution_x; ++j2)
      {
        float x2 =  static_cast<float>(j2)/static_cast<float>(resolution_x);
        float y2 =  static_cast<float>(i2)/static_cast<float>(resolution_x);
        float dist_squared = (u_x - x2)*(u_x - x2) + (u_y - y2)*(u_y - y2);

        if(data2[j2+i2*resolution_x] >= 0.0f && dist_squared < smallest_dist_squared2)
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
 * function3() - controlls the bounding box insied of which space-time blending took place
 */
float shader::ColorShader::function3(ngl::Vec2 uv, float time)
{
#ifdef USE_CONE_F3

  float t_min = -10.0f;
  float t_max =  10.0f;
  float K, X_C, Y_C, R, ddd, f3;

  K = (time - t_min) / (t_max - t_min);

  X_C = center1[0] + K * (center2[0] - center1[0]);
  Y_C = center1[1] + K * (center2[1] - center1[1]);

  R  = r1 + K * (r2 - r1);

  ddd = R * R - ( uv.m_x - X_C )*( uv.m_x - X_C ) -
                ( uv.m_y - Y_C )*( uv.m_y - Y_C );

  f3 = INTERSECT( INTERSECT( ddd, 10.0f + time ), ( 10.0f - time ) );

#endif //USE_CONE_F3

#ifdef USE_PYRAMID_F3

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

#ifndef USE_CONE_F3
#ifndef USE_PYRAMID_F3
  float f3 = INTERSECT(time + 10.0f, 10.0f - time);
#endif
#endif // if not defined(USE_CONE_F3) || not defined(USE_PYRAMID_F3)

  return f3;
}


/*
 * functions below are providing color conversion rules between RGB/HSV color spaces;
 */
#ifdef USE_OPENMP
#pragma omp declare simd
#endif
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

#ifdef USE_OPENMP
#pragma omp declare simd
#endif
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

#ifdef USE_OPENMP
#pragma omp declare simd
#endif
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

#ifdef USE_OPENMP
#pragma omp declare simd
#endif
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

#ifdef USE_OPENMP
#pragma omp declare simd
#endif
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

#ifdef USE_OPENMP
#pragma omp declare simd
#endif
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
