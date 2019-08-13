#include "include/metamorphosis.h"

#include <iostream>
#include <chrono>
#include <ctime>

#ifdef USE_OPENMP
  #include <omp.h>
#endif

namespace metamorphosis {

metamorphosis2D::metamorphosis2D( cv::Mat inputImg, cv::Mat targImg, ColorMode colorMode, DrawMode drawMode,
                                  int totalFrames, float scale, bool autoMode)
{
    if( inputImg.cols == targImg.cols && inputImg.rows == targImg.rows )
    {
        if( scale > 0.0f && scale != 1.0f )
        {
            scaleImage( &inputShape, inputImg, scale, cv::INTER_NEAREST );
            scaleImage( &targetShape, targImg, scale, cv::INTER_NEAREST );
        }
        else {
            inputImg.copyTo( inputShape );
            targImg.copyTo( targetShape );
        }

        inputImBuf   = inputShape.data;
        targImBuf    = targetShape.data;
        inputChs     = inputShape.channels();
        targChs      = targetShape.channels();
        stepInputImg = inputShape.step;
        stepTargIm   = targetShape.step;

        resultImg   = cv::Mat( inputShape.cols, inputShape.rows, inputShape.type());
        resultImBuf = resultImg.data;
        stepResImg  = resultImg.step;
        resChs      = resultImg.channels();

        curColorMode = colorMode;
        curDrawMode  = drawMode;
        frameCount   = totalFrames;
        curFrame     = 0;
        resX         = inputShape.cols;
        resY         = inputShape.rows;

        finScale = scale;

        a0 = 1.0f;
        a1 = 1.0f;
        a2 = 1.0f;
        a3 = 1.0f;

        //setting up coeffcients for smoothing half-cylinders
        //input shape:  b0_1, b1_1, b2_1, b3_1
        //target shape: b0_2, b1_2, b2_2, b3_2
#if not defined(USE_CONE_F3)
#if not defined(USE_PYRAMID_F3)
        imgProc1 = std::make_shared<imageProc>( &inputShape, &targetShape, false );
        b0_1 = -0.1f;       //-0.3f (-0.1f for 3circle example)
        b0_2 = -0.5f;       //-0.5f
#endif
#endif

#if defined(USE_CONE_F3)
        imgProc1 = std::make_shared<imageProc>( &inputShape, &targetShape );
        b0_1 = -0.8f;
        b0_2 = -0.2f; // -0.5f;
#elif defined(USE_PYRAMID_F3)
        imgProc1 = std::make_shared<imageProc>( &inputShape, &targetShape, false );
        imgProc2 = std::make_shared<imageProc>( &inputShape, &targetShape );
        b0_1 = -0.3f;
        b0_2 = -0.5f;
#endif

        b1_1 = 1.0f; b1_2 = 1.0f;
        b2_1 = 1.0f; b2_2 = 1.0f;
        b3_1 = 1.0f; b3_2 = 1.0f;

#ifdef USE_AFFINE_TRANSFORMATIONS
        float d_r1, d_r2;
        imgProc2 = std::make_shared<imageProc>( &inputShape, &targetShape, false );
        imgProc2.get()->getMaxCircles(&d_r1, &d_r2, &rec1_c, &rec2_c );
        shiftX = rec2_c.x - rec1_c.x;
        shiftY = rec2_c.y - rec1_c.y;
#endif
        DT = std::make_shared<distanceTransform>( &inputShape, &targetShape, 0.0f );

        std::vector<float> ddt1 = DT.get()->getSDDT1();
        std::vector<float> ddt2 = DT.get()->getSDDT2();

        /*F1.resize( resX * resY ); F2.resize( resX * resY );
        DT.get()->smoothDistanceTransformField( &ddt1, &F1 );
        DT.get()->smoothDistanceTransformField( &ddt2, &F2 );*/

        F1 = ddt1;
        F2 = ddt2;

        colInter = std::make_shared<colourInter>( inputImg, targImg, F1, F2, resX, resY, curDrawMode, curColorMode, 10 );

#ifndef USE_AFFINE_TRANSFORMATIONS
        if(autoMode)
            setAutoSTBcoefficients();
#else
        std::cout << "\nUsing Affine translation! No manual mode for STB coefficiients!" << std::endl;
#endif
    }
    else {
        std::cerr <<" ERROR: input and target images have different dimensions <exit>!" << std::endl;
        exit(-1);
    }
}

void metamorphosis2D::calcAffineTransformedDT( float shiftX, float shiftY )
{
    targetShape = imgProc2.get()->getAffineTransformedImage( &targetShape, &affine_mat, shiftX, shiftY );
    targImBuf   = targetShape.data;

    imgProc1 = std::make_shared<imageProc>( &inputShape, &targetShape, false );
    setAutoSTBcoefficients();

    F2_m.clear();
    F2_m.resize( F2.size() );
    float *F2buf = F2_m.data();
    cv::Mat affine_m = affine_mat;
    float result = 0.0f;
    int res_x = resX, res_y = resY;
    float u, v;

    for(int y = 0; y < res_y; ++y)
    {
        for(int x = 0; x < res_x; ++x)
        {
            u = static_cast<float>(x)/static_cast<float>(res_x);
            v = static_cast<float>(y)/static_cast<float>(res_y);

            DT.get()->inverseMappingResult( &result, F2, u, v, affine_mat );
            F2buf[x+y*resX] = result;
        }
    }
}

void metamorphosis2D::calculateMetamorphosis()
{
    auto start1 = std::chrono::high_resolution_clock::now();

#if defined(USE_AFFINE_TRANSFORMATIONS) || defined(USE_CONE_F3) || defined(USE_PYRAMID_F3)

#if defined(USE_CONE_F3)
    imgProc1.get()->getMaxCircles(&rad1, &rad2, &cen1, &cen2);
    convertToUV( &rad1, &rad2 );
#endif //USE_CONE_F3

#ifdef USE_PYRAMID_F3
    imgProc2.get()->getMaxRectangles( &rec1, &rec2 );
    imgProc2.get()->getRectangleCenters( &rec1_c, &rec2_c );
    convertToUV( &rec1_c.x, &rec1_c.y );
    convertToUV( &rec2_c.x, &rec2_c.y );
#endif //USE_PYRAMID_F3

#ifdef USE_AFFINE_TRANSFORMATIONS
    float stX = (rec1_c.x - rec2_c.x)*resX;
    float stY = (rec1_c.y - rec2_c.y)*resY;
    calcAffineTransformedDT( stX, stY );
#endif
#endif

    while( curFrame < frameCount )
    {
        auto start2 = std::chrono::high_resolution_clock::now();

        glm::vec4 color( 0.0f, 0.0f, 0.f, 1.0f );
        for(int y = 0; y < resY; y++ )
        {
            for(int x = 0; x < resX; x++ )
            {
#ifndef USE_CROSS_DISSOLVE
                color = obtainResultingColor( glm::vec2( x, y ), curFrame );
                resultImBuf[x*resChs  +y*stepResImg] = color.b * 255;
                resultImBuf[x*resChs+1+y*stepResImg] = color.g * 255;
                resultImBuf[x*resChs+2+y*stepResImg] = color.r * 255;
                resultImBuf[x*resChs+3+y*stepResImg] = color.a * 255;
#else
                float time = static_cast<float>(curFrame) / static_cast<float>(frameCount);

                resultImBuf[x*resChs  +y*stepResImg] = (1.0f-time)*inputImBuf[x*inputChs   + y*stepInputImg] +
                                                             time * targImBuf[x*targChs    + y*stepTargIm];
                resultImBuf[x*resChs+1+y*stepResImg] = (1.0f-time)*inputImBuf[x*inputChs+1 + y*stepInputImg] +
                                                             time * targImBuf[x*targChs+1  + y*stepTargIm];
                resultImBuf[x*resChs+2+y*stepResImg] = (1.0f-time)*inputImBuf[x*inputChs+2 + y*stepInputImg] +
                                                             time * targImBuf[x*targChs+2  + y*stepTargIm];
                resultImBuf[x*resChs+3+y*stepResImg] = (1.0f-time)*inputImBuf[x*inputChs+3 + y*stepInputImg] +
                                                             time * targImBuf[x*targChs+3  + y*stepTargIm];
#endif
            }
        }

        cv::Mat dst;
        if( finScale > 0.0f && finScale != 1.0f )
        {
            scaleImage( &dst, resultImg, 1.0f/finScale, cv::INTER_AREA );
        }
        else {
            dst = cv::Mat( resX, resY, resultImg.type());
            resultImg.copyTo( dst );
        }

        if( !background.empty() )
            useImageBackground( &dst, dst );

        std::string fullPath;
        if( !outputPath.empty() )
        {
            fullPath = outputPath + std::to_string( curFrame ) + ".png";
        }
        else {
            fullPath = "/OUT/frame" + std::to_string( curFrame ) + ".png";
        }

        cv::imwrite( fullPath, dst );

        auto end2 = std::chrono::high_resolution_clock::now();
        auto dur1 = std::chrono::duration_cast<std::chrono::duration<float>>(end2 - start2);

        std::cout << "image_"       << curFrame     << " is saved." << std::endl;
        std::cout << "spent time: " << dur1.count() << " seconds"   << std::endl;
        time.push_back( dur1.count() );

        curFrame++;
    } //while( curFrame < frameCount )

    auto end1 = std::chrono::high_resolution_clock::now();
    auto dur2 = std::chrono::duration_cast<std::chrono::duration<double>>(end1 - start1);

    std::cout << "\n\ntotal_time [sec]:"    << dur2.count() << std::endl;
    std::cout << "min_time [FPS]: "     << 1.0f / ( *std::minmax_element(time.begin(), time.end()).first )  << std::endl;
    std::cout << "max_time [FPS]: "     << 1.0f / ( *std::minmax_element(time.begin(), time.end()).second ) << std::endl;
    std::cout << "average_time [FPS]: " << frameCount / dur2.count() << std::endl;
}

void metamorphosis2D::useImageBackground(cv::Mat *dst, cv::Mat curImg )
{
    if(background.empty())
    {
        std::cerr << "ERROR: failed to load background image [--back]! " << std::endl;
        return;
    }

    unsigned char *dst_buf      = dst->data;
    unsigned char *back_img_buf = background.data;
    unsigned char *src_buf      = curImg.data;
    int b_ch = background.channels();
    int d_ch = dst->channels();
    int s_ch = curImg.channels();
    size_t b_step = background.step;
    size_t d_step = dst->step;
    size_t s_step = curImg.step;
    int res_x = curImg.cols, res_y = curImg.rows;

#ifdef USE_OPENMP
#pragma omp parallel shared(res_x, res_y, s_ch, s_step, d_ch, d_step, s_ch, s_step, b_ch, b_step, dst_buf, back_img_buf, src_buf)
{
#pragma omp for simd schedule(static,5)
#endif
    for(int y = 0; y < res_y; ++y)
    {
        for(int x = 0; x < res_x; ++x)
        {
            if(src_buf[x*s_ch+3+y*s_step] == 0)
            {
                dst_buf[x*d_ch  +y*d_step] = back_img_buf[x*b_ch  +y*b_step];
                dst_buf[x*d_ch+1+y*d_step] = back_img_buf[x*b_ch+1+y*b_step];
                dst_buf[x*d_ch+2+y*d_step] = back_img_buf[x*b_ch+2+y*b_step];
                dst_buf[x*d_ch+3+y*d_step] = back_img_buf[x*b_ch+3+y*b_step];
            }
            else
            {
                dst_buf[x*d_ch  +y*d_step] = src_buf[x*s_ch  +y*s_step];
                dst_buf[x*d_ch+1+y*d_step] = src_buf[x*s_ch+1+y*s_step];
                dst_buf[x*d_ch+2+y*d_step] = src_buf[x*s_ch+2+y*s_step];
                dst_buf[x*d_ch+3+y*d_step] = src_buf[x*s_ch+3+y*s_step];
            }
        }
    }
#ifdef USE_OPENMP
}
#endif

}

void metamorphosis2D::scaleImage( cv::Mat *dst, cv::Mat src, float scale, int cv_interpolation )
{
    *dst  = cv::Mat( static_cast<int>(src.cols/scale), static_cast<int>(src.rows/scale), src.type());
    if(cv_interpolation == cv::INTER_NEAREST)
        cv::resize( src, *dst, dst->size(), 0, 0, cv_interpolation );
    else {
        cv::resize( src, *dst, dst->size(), cv_interpolation );
    }
}

void metamorphosis2D::setAutoSTBcoefficients()
{
    imgProc1.get()->getMaxCircles( &rad1, &rad2, &cen1, &cen2 );
    convertToUV( &rad1, &rad2 );

    calcA1A2coefficient();
    calcA0coefficient();
    caclA3Coefficient();

    std::cout << " a1 = a2 = " << a1 << std::endl;
    std::cout << "      a0 = " << a0 << std::endl;
    std::cout << "      a3 = " << a3 << std::endl;
}

void metamorphosis2D::calcA1A2coefficient()
{
    //****************************************************************************************
    //STEP I: find middle points of the line connected centers of two circles for estimating a1
    float dist = std::sqrt(( cen2.x - cen1.x )*( cen2.x - cen1.x ) +
                           ( cen2.y - cen1.y )*( cen2.y - cen1.y ));
    if( dist < 0.02f)
    {
        a1 = 1.0f;
        a2 = 1.0f;
    }
    else {
        //calculating glm::vec2 K1
        //coeffs for K1x

        glm::vec2 c1_2( cen1.x*cen1.x, cen1.y*cen1.y );
        glm::vec2 c1_3( c1_2.x * cen1.x, c1_2.y * cen1.y );
        glm::vec2 c1_4( c1_2.x * c1_2.x, c1_2.y * c1_2.y );

        glm::vec2 c2_2( cen2.x*cen2.x, cen2.y*cen2.y );
        glm::vec2 c2_3( c2_2.x * cen2.x, c2_2.y * cen2.y );
        glm::vec2 c2_4( c2_2.x * c2_2.x, c2_2.y * c2_2.y );

        glm::vec2 dCent12( cen2.x - cen1.x, cen2.y - cen1.y );

        float k = 1.0f / ( dCent12.x*dCent12.x + dCent12.y*dCent12.y);

        float k1_1 = c1_3.x        - 2.0f*c1_2.x*cen2.x        + cen1.x*c2_2.x +
                     cen1.x*c1_2.y - 2.0f*cen1.x*cen1.y*cen2.y + cen1.x*c2_2.y;

        float k1_2 = c1_4.x - 4.0f*c1_3.x*cen2.x + 6.0f*c1_2.x*c2_2.x        - 4.0f*cen1.x*c2_3.x +
                     c2_4.x + c1_2.x*c1_2.y      - 2.0f*cen1.x*cen2.x*c1_2.y + c2_2.x*c1_2.y -
                     2.0f*c1_2.x*cen1.y*cen2.y   + 4.0f*cen1.x*cen2.x*cen1.y*cen2.y -
                     2.0f*c2_2.x*cen1.y*cen2.y   + c1_2.x*c2_2.y -
                     2.0f*cen1.x*cen2.x*c2_2.y   + c2_2.x*c2_2.y;

        //coeffs for K1y
        float k1_3 = c1_3.x*cen1.y             - 2.0f*c1_2.x*cen2.x*cen1.y +
                     c2_2.x*cen1.x*cen1.y      + cen1.x* c1_3.y -
                     c1_3.x*cen2.y             + 2.0f*c1_2.x*cen2.x*cen2.y -
                     c2_2.x*cen1.x*cen2.y      - 3.0f*c1_2.y*cen1.x*cen2.y +
                     3.0f*cen1.x*cen1.y*c2_2.y - cen1.x*c2_3.y;

        float K1x_a = k * ( k1_1 - rad1 * std::sqrt( k1_2 ));
        float K1y_a = ( -1.0f/dCent12.x ) * ( -cen2.x*cen1.y + cen1.x*cen2.y +
                      k * ( k1_3 + dCent12.y * rad1 * std::sqrt( k1_2 )));

        float K1x_b = k * ( k1_1 + rad1 * std::sqrt( k1_2 ));
        float K1y_b = ( -1.0f/dCent12.x) * ( -cen2.x*cen1.y + cen1.x*cen2.y +
                      k * ( k1_3 - dCent12.y * rad1 * std::sqrt( k1_2 )));

        //****************************************************************************************
        //calculating glm::vec2 K2
        //coeffs for K2x1
        float k2_1 = c1_2.x*cen2.x - 2.0f*cen1.x*c2_2.x + c2_3.x +
                     cen2.x*c1_2.y - 2.0f*cen2.x*cen1.y*cen2.y + cen2.x*c2_2.y;
        float k2_2 = k1_2;

        //coeffs for K2y
        float k2_3 = c1_2.x*cen2.x*cen1.y      - 2.0f*cen1.x*c2_2.x*cen1.y +
                     c2_3.x*cen1.y             + cen2.x* c1_3.y -
                     c1_2.x*cen2.x*cen2.y      + 2.0f*cen1.x*c2_2.x*cen2.y -
                     c2_3.x*cen2.y             - 3.0f*cen2.x*c1_2.y*cen2.y +
                     3.0f*cen2.x*cen1.y*c2_2.y - cen2.x*c2_3.y;

        float K2x_a = k * ( k2_1 - rad2 * std::sqrt( k2_2 ));
        float K2y_a = ( -1.0f/dCent12.x) * ( -cen2.x*cen1.y + cen1.x*cen2.y +
                      k * ( k2_3 + dCent12.y * rad2 * std::sqrt( k2_2 )));

        float K2x_b = k * ( k2_1 + rad2 * std::sqrt( k2_2 ));
        float K2y_b = ( -1.0f/dCent12.x ) * ( -cen2.x*cen1.y + cen1.x*cen2.y +
                      k * ( k2_3 - dCent12.y * rad2 * std::sqrt( k2_2 )));

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

        //calculating f1 and f2 in these glm::vec2s
        float f1_k = rad1 - std::sqrt((K2x - cen1.x)*(K2x - cen1.x)) -
                     std::sqrt((K2y - cen1.y)*(K2y - cen1.y));
        float f2_k = rad2 - std::sqrt((K1x - cen2.x)*(K1x - cen2.x)) -
                     std::sqrt((K1y - cen2.y)*(K1y - cen2.y));

        a1 = std::sqrt( (f1_k*f1_k + f2_k*f2_k) / norm );
        a2 = a1;
    }
}

void metamorphosis2D::calcA0coefficient()
{
    float x_min, x_max;
    float cx_min = std::min(cen1.x, cen2.x);

    if( cx_min == cen1.x )
    {
        x_min = cx_min - rad1;
        x_max = cen2.x + rad2;
    }
    else
    {
        x_min = cx_min - rad2;
        x_max = cen1.x + rad1;
    }

    float y_min, y_max;
    float cy_min = std::min(cen1.y, cen2.y);
    if(cy_min == cen1.y )
    {
        y_min = cy_min - rad1;
        y_max = cen2.y + rad2;
    }
    else
    {
        y_min = cy_min - rad2;
        y_max = cen1.y + rad1;
    }

    if(x_min < 0.0f) x_min = 0.0f;
    if(x_max > 1.0f) x_max = 1.0f;
    if(y_min < 0.0f) y_min = 0.0f;
    if(y_max > 1.0f) y_max = 1.0f;

    //intervals for sqrt[x*x-2xx1+x1*x1] and sqrt[y*y-2yx1+y1*y1]; sqrt[x*x-2xx2+x2*x2] and sqrt[y*y-2yx2+y2*y2]
    float x1_min = x_min*x_max - 2.0f*cen1.x*x_max + cen1.x*cen1.x;
    float y1_min = y_min*y_max - 2.0f*cen1.y*y_max + cen1.y*cen1.y;
    float x2_min = x_min*x_max - 2.0f*cen2.x*x_max + cen2.x*cen2.x;
    float y2_min = y_min*y_max - 2.0f*cen2.y*y_max + cen2.y*cen2.y;

    if(x1_min < 0.0f) x1_min = 0.0f;
    if(y1_min < 0.0f) y1_min = 0.0f;
    if(x2_min < 0.0f) x2_min = 0.0f;
    if(y2_min < 0.0f) y2_min = 0.0f;

    float x1_max = x_max*x_max - 2.0f*cen1.x*x_min + cen1.x*cen1.x;
    float y1_max = y_max*y_max - 2.0f*cen1.y*y_min + cen1.y*cen1.y;
    float x2_max = x_max*x_max - 2.0f*cen2.x*x_min + cen2.x*cen2.x;
    float y2_max = y_max*y_max - 2.0f*cen2.y*y_min + cen2.y*cen2.y;

    //intervals for function f1 and f2 represented as circle equations
    glm::vec2 fun1( std::sqrt(x1_min) + std::sqrt(y1_min) - rad1,
                    std::sqrt(x1_max) + std::sqrt(y1_max) - rad1 );
    glm::vec2 fun2( std::sqrt(x2_min) + std::sqrt(y2_min) - rad2,
                    std::sqrt(x2_max) + std::sqrt(y2_max) - rad2 );

    //f1*f1 and f2*f2 check if the min value is positive, otherwise it is 0.0
    float fun11_min = fun1.x*fun1.y;
    float fun22_min = fun2.x*fun2.y;

    if( fun11_min < 0.0f ) fun11_min = 0.0f;
    if( fun22_min < 0.0f ) fun22_min = 0.0f;

    glm::vec2 fun11( fun11_min, fun1.y*fun1.y );
    glm::vec2 fun22( fun22_min, fun2.y*fun2.y );

    //checking whether sum under the square root is positive: sqrt(f1*f1 + f2*f2)

    glm::vec2 f11f22_sum( fun11.x + fun22.x, fun11.y + fun22.y );
    glm::vec2 SqrF12F22_sum( std::sqrt(f11f22_sum.x), std::sqrt(f11f22_sum.y) );

    //calculating final estimation interval for a0
    glm::vec2 f1f2_sum( fun1.x + fun2.x , fun1.y + fun2.y );

    if( check2CircleCrosssection() && ((rad2-rad1)/std::sqrt( (cen2.x - cen1.x)*(cen2.x - cen1.x)
                                                            + (cen2.y - cen1.y)*(cen2.y - cen1.y)) < 1.0f ))
    {
        float dy = ( cen2.y - cen1.y );
        float dx = ( cen2.x - cen1.x );
        float dr = ( rad2 - rad1 );
        float d  = std::sqrt( dx*dx + dy*dy );

        float X = dx / d;
        float Y = dy / d;
        float R = dr / d;

        //for top tangential line
        float A1 = R*X - Y*std::sqrt( 1.0f - R*R );
        float B1 = R*Y + X*std::sqrt( 1.0f - R*R );
        float C1 = rad1 - ( A1*cen1.x + B1*cen1.y );

        //for bottom tangential line
        float A2 = R*X + Y*std::sqrt( 1.0f - R*R );
        float B2 = R*Y - X*std::sqrt( 1.0f - R*R );
        float C2 = rad1 - ( A2*cen1.x + B2*cen1.y );

        glm::vec2 AA( f1f2_sum.x + SqrF12F22_sum.x, f1f2_sum.y + SqrF12F22_sum.y );
        glm::vec2 BB( 1.0f + (1.0f/(a1*a1))*f11f22_sum.x, 1.0f + (1.0f/(a1*a1))*f11f22_sum.y );
        glm::vec2 CC1( A1*x_min + B1*y_min + C1, A1*x_max + B1*y_max + C1 );
        glm::vec2 CC2( A2*x_min + B2*y_min + C2, A2*x_max + B2*y_max + C2 );

        glm::vec2 DD1( CC1.x - AA.y, CC1.y - AA.x );
        glm::vec2 DD2( CC2.x - AA.y, CC2.y - AA.x );

        //found bottom interval for a0
        glm::vec2 a0_bot( min4(DD2.x*BB.x, DD2.x*BB.y, DD2.y*BB.x, DD2.y*BB.y),
                          max4(DD2.x*BB.x, DD2.x*BB.y, DD2.y*BB.x, DD2.y*BB.y) );

        glm::vec2 a0_top( min4(DD1.x*BB.x, DD1.x*BB.y, DD1.y*BB.x, DD1.y*BB.y),
                          max4(DD1.x*BB.x, DD1.x*BB.y, DD1.y*BB.x, DD1.y*BB.y) );

        float a0_min, a0_max;
        if(a0_bot.x < 0.0f && a0_top.x < 0.0f && a0_bot.y > 0.0f)
            a0_min = a0_bot.y;
        else
            a0_min = a0_bot.x;

        a0_max = a0_top.y;

        glm::vec2 a0_int(a0_min, a0_max);

        std::cout << "bottom interval for a0 [ " << a0_bot.x << " , " << a0_bot.y << " ]" << std::endl;
        std::cout << "top    interval for a0 [ " << a0_top.x << " , " << a0_top.y << " ]" << std::endl;
        std::cout << "final  interval for a0 [ " << a0_min   << " , " << a0_max   << " ]" << std::endl;
        a0 = a0_max;
    }
    else
    {
        float cx, cy, r_uv;
        if( std::max(rad1, rad2) == rad1 )
        {
            cx = cen1.x;
            cy = cen1.y;
            r_uv = rad1;
        }
        else
        {
            cx = cen2.x;
            cy = cen2.y;
            r_uv = rad2;
        }

        glm::vec2 AA( f1f2_sum.x + SqrF12F22_sum.x, f1f2_sum.y + SqrF12F22_sum.y );
        glm::vec2 BB( 1.0f + (1.0f/(a1*a1))*f11f22_sum.x, 1.0f + (1.0f/(a1*a1))*f11f22_sum.y );
        glm::vec2 CC( (x_min - cx)*(x_max - cx) + (y_min - cy)*(y_max - cy) - r_uv*r_uv,
                      (x_max - cx)*(x_max - cx) + (y_max - cy)*(y_max - cy) - r_uv*r_uv );

        glm::vec2 DD( CC.x - AA.y, CC.y - AA.x );

        glm::vec2 a0_int( min4(DD.x*BB.x, DD.y*BB.x, DD.x*BB.y, DD.y*BB.y),
                          max4(DD.x*BB.x, DD.y*BB.x, DD.x*BB.y, DD.y*BB.y) );

        std::cout << "final  interval for a0 [ " << a0_int.x    << " , " << a0_int.y    << " ]" << std::endl;
        a0 = a0_int.y;

        glm::vec2 dist( cen2.x - cen1.x, cen2.y - cen1.y );
        minDist = std::sqrt( dist.x*dist.x + dist.y*dist.y );
    }
}

void metamorphosis2D::caclA3Coefficient()
{
#ifdef USE_CONE_F3
    //I manually found the proper value for one case and want to generalize it
    // for other cases and stick it to geometrical parameter of the shape
    a3 = std::min(rad1, rad2)*0.03126f;
#elif defined (USE_PYRAMID_F3)
    a3 = std::min(rad1, rad2)*0.13336f;
#endif
}

bool metamorphosis2D::check2CircleCrosssection()
{
    glm::vec2 c1_2( cen1.x*cen1.x, cen1.y*cen1.y );
    glm::vec2 c2_2( cen2.x*cen2.x, cen2.y*cen2.y );
    glm::vec2 deltaC12( cen1.x - cen2.x, cen1.y - cen2.y);
    glm::vec2 deltaC12_2( deltaC12.x*deltaC12.x, deltaC12.y*deltaC12.y );

    float coef1 = c1_2.x + c1_2.y - rad1*rad1;
    float coef2 = c2_2.x + c2_2.y - rad2*rad2;

    float c1 = 4.0f*( deltaC12_2.x + deltaC12_2.y );
    float c2 = (coef1 - coef2)*(coef1 - coef2) + c1_2.y*(4.0f*coef2 - 8.0f*coef1) +
               4.0f*cen1.y*cen2.y*(3.0f*coef1 - coef2) - 4.0f*coef1*c2_2.y;
    float c3 = -4.0f*coef1*deltaC12.x - 4.0f*coef2*deltaC12.x - 8.0f*cen2.x*(c1_2.y - cen1.y*cen2.y)
                                                              - 8.0f*cen1.x*(c2_2.y - cen1.y*cen2.y);
    float diskr = -4.0f*c1*c2 + c3*c3;

    if(diskr < 0.0f && std::sqrt( deltaC12_2.x + deltaC12_2.y ) < std::min(rad1, rad2)/2.0f )
        return false;
    else if (diskr < 0.01f)
        return false;
    else if (diskr == 0.0f)
        return false;
    else
        return true;
}

float metamorphosis2D::smoothHalfCylinder1(float functionVal, float time)
{
    float f1 = functionVal;
    float f2 = -time;
    float f3 =  time + 5.0f;

    float r1 = (f1/b1_1)*(f1/b1_1) + (f2/b2_1)*(f2/b2_1);
    float r2 = 0.0f;

    if( f3 > 0.0f )
    {
      r2 = (f3/b3_1) * (f3/b3_1);
    }

    float rr = 0.0f;
    if( r1 > 0.0f )
    {
      rr = r1 / (r1 + r2);
    }

    float d = 0.0f;
    if( rr < 1.0f )
    {
      d = b0_1 * (1.0f - rr)*(1.0f - rr)*(1.0f - rr) / (1.0f + rr);
    }

    float blending_result = intersectFunctions( f1, f2 ) + d;
    return blending_result;
}

float metamorphosis2D::smoothHalfCylinder2(float functionVal, float time)
{
    float f1 = functionVal;
    float f2 = time - 1.0f;
    float f3 = 5.0f - time;

    float r1 = (f1/b1_2)*(f1/b1_2) + (f2/b2_2)*(f2/b2_2);
    float r2 = 0.0f;

    if( f3 > 0.0f )
    {
      r2 = (f3/b3_2) * (f3/b3_2);
    }

    float rr = 0.0f;
    if( r1 > 0.0f )
    {
      rr = r1 / (r1 + r2);
    }

    float d = 0.0f;
    if( rr < 1.0f )
    {
      d = b0_2 * (1.0f - rr)*(1.0f - rr)*(1.0f - rr) / (1.0f + rr);
    }

    float blending_result = intersectFunctions( f1, f2 ) + d;
    return blending_result;
}

float metamorphosis2D::obtainF3Values(glm::vec2 uv, float time)
{
#ifdef USE_CONE_F3

    float t_min = -10.0f;
    float t_max =  10.0f;
    float K, X_C, Y_C, R, ddd, f3;

    K = ( time - t_min ) / ( t_max - t_min );

    X_C = cen1.x + K * ( cen2.x - cen1.x );
    Y_C = cen1.y + K * ( cen2.y - cen1.y );

    R   = rad1 + K * (rad2 - rad1);
    ddd = R * R - ( uv.x - X_C )*( uv.x - X_C ) - ( uv.y - Y_C )*( uv.y - Y_C );

    f3 = intersectFunctions( intersectFunctions( ddd, 10.0f + time ), ( 10.0f - time ) );

#endif //USE_CONE_F3

#ifdef USE_PYRAMID_F3

    float t_min = -10.0f;
    float t_max =  10.0f;
    float K = ( time - t_min ) / ( t_max - t_min );

    float rec_w = rec1.width/(float)resX  + K * ( rec2.width/(float)resX  - rec1.width/(float)resX  );
    float rec_h = rec1.height/(float)resY + K * ( rec2.height/(float)resY - rec1.height/(float)resY );

    float X_C = rec1_c.x + K * ( rec2_c.x - rec1_c.x );
    float Y_C = rec1_c.y + K * ( rec2_c.y - rec1_c.y );

    float ddd = intersectFunctions( rec_w - std::abs( uv.x - X_C ), rec_h - std::abs( uv.y - Y_C ));

    float f3  = intersectFunctions( intersectFunctions( ddd, 10.0f + time ), ( 10.0f - time ));

#endif //USE_PYRAMID_F3

#ifndef USE_CONE_F3
#ifndef USE_PYRAMID_F3
  float f3 = intersectFunctions(time + 10.0f, 10.0f - time);
#endif
#endif // if not defined(USE_CONE_F3) || not defined(USE_PYRAMID_F3)

  return f3;
}

glm::vec4 metamorphosis2D::obtainResultingColor(glm::vec2 coord, int frameNumber)
{
    float time = 0.0;
    convertToUV( &coord.x, &coord.y );

    float t = static_cast<float>( frameNumber ) / static_cast<float>( frameCount );
    glm::vec4 color;

    if( t <= 0.01f )
    {
       time = -10.0f;
       color = obtainBlendingResult( coord, time);
    }
    else
    {
       time = ( 20.0f / 0.36f ) * ( t - 0.01f ) * ( t - 0.01f ) - 10.0f;
       color = obtainBlendingResult( coord, time);
    }

    return color;
}

glm::vec4 metamorphosis2D::obtainBlendingResult(glm::vec2 uv, float time)
{
    float fun1, fun2;
    float f1, f2;

#ifdef USE_SMOOTHED_CYLINDERS
    fun1 = F1[static_cast<size_t>( uv.x*resX ) + static_cast<size_t>( uv.y*resY*resX )];
#ifdef USE_AFFINE_TRANSFORMATIONS
    fun2 = F2_m[static_cast<size_t>( uv.x*resX ) + static_cast<size_t>( uv.y*resY*resX )];
#else
    fun2 = F2[static_cast<size_t>( uv.x*resX ) + static_cast<size_t>( uv.y*resY*resX )];
#endif
    f1 = smoothHalfCylinder1( fun1, time );
    f2 = smoothHalfCylinder2( fun2, time );
#else
    fun1 = F1[static_cast<int>( uv.x*resX ) + static_cast<int>( uv.y*resY*resX )];
    fun2 = F2[static_cast<int>( uv.x*resX ) + static_cast<int>( uv.y*resY*resX )];

    f1 = intersectFunctions( fun1, -time );
    f2 = intersectFunctions( fun2, ( time - 1.0f ));
#endif //USE_SMOOTHED_CYLINDERS

    float f3 = obtainF3Values( uv, time );

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

    float blending_result = unionFunctions(f1, f2) + d;

    glm::vec4 objCol(0.0, 0.0, 0.0, 1.0);
    if(blending_result >= 0.0f)
    {
        if(f1 < 0.0f && f2 < 0.0f)
            objCol = colInter.get()->computeSTTI_noCorrespondences( f1, f2, fun1, fun2, uv );
        else
        {
            if(f1 >= 0.0f)
                objCol = colInter.get()->getPixelColor( inputImBuf, stepInputImg, inputChs, uv );
            if(f2 >= 0.0f)
                objCol = colInter.get()->getPixelColor( targImBuf, stepTargIm, targChs, uv );
        }
    }

    colInter.get()->convertToRGB( &objCol.r, &objCol.g, &objCol.b );
    colInter.get()->clampColor( &objCol );

#ifdef USE_TRANSPARENT_BACKBGROUND
    if( objCol == glm::vec4(0.0, 0.0, 0.0, 1.0))
        objCol = glm::vec4(0.0, 0.0, 0.0, 0.0);
#endif

    return objCol;
}

}//namespace metamorphosis
