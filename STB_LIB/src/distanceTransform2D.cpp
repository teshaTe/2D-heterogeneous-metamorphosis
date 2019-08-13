#include "include/distanceTransform2D.h"

#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

#include <iostream>

#ifdef USE_OPENMP
    #include <omp.h>
#endif

namespace metamorphosis {

Point::Point(float dx, float dy) : x(dx), y(dy) { }
float Point::square_distance() { return (std::pow(x, 2.0) + std::pow(y, 2.0)); }

distanceTransform::distanceTransform(cv::Mat *inputImg, cv::Mat *targetImg, int b_sh): bShift( b_sh ),
                                                                                       resX( inputImg->cols+bShift ),
                                                                                       resY( inputImg->rows+bShift ),
                                                                                       INF( calculateINF() ),
                                                                                       INSIDE( INF, INF ),
                                                                                       OUTSIDE( 0.0f, 0.0f ),
                                                                                       grid1( resX*resY, INSIDE ),
                                                                                       grid2( resX*resY, INSIDE ),
                                                                                       SDDT1( resX*resY, 0.0f ),
                                                                                       SDDT2( resX*resY, 0.0f )
{
    cv::Mat dst1, dst2;

    if(b_sh == 0)
    {
       inputImg->copyTo(dst1);
       targetImg->copyTo(dst2);
    }
    else
    {
       cv::copyMakeBorder( *inputImg,  dst1, 0, b_sh, 0, b_sh, cv::BORDER_REPLICATE );
       cv::copyMakeBorder( *targetImg, dst2, 0, b_sh, 0, b_sh, cv::BORDER_REPLICATE );
    }

    generateDistanceTransform( SDDT1, &dst1 );
    generateDistanceTransform( SDDT2, &dst2 );
}

distanceTransform::distanceTransform(cv::Mat *img, int b_sh): bShift( b_sh ),
                                                              resX( img->cols+bShift ),
                                                              resY( img->rows+bShift ),
                                                              INF( calculateINF() ),
                                                              INSIDE( INF, INF ),
                                                              OUTSIDE( 0.0f, 0.0f ),
                                                              grid1( resX*resY, INSIDE ),
                                                              grid2( resX*resY, INSIDE ),
                                                              SDDT1( resX*resY, 0.0f )
{
    cv::Mat dst;
    if(b_sh == 0)
        img->copyTo(dst);
    else
        cv::copyMakeBorder( *img, dst, 0, b_sh, 0, b_sh, cv::BORDER_REPLICATE );

    generateDistanceTransform( SDDT1, &dst );
}

cv::Mat distanceTransform::binariseImage(cv::Mat *src)
{
    cv::Mat srcGray, binImg;
    cv::cvtColor( *src, srcGray, cv::COLOR_BGR2GRAY, 1);
    cv::threshold( srcGray, binImg, 0, 255, cv::THRESH_BINARY );

    return binImg;
}

void distanceTransform::generateDistanceTransform(std::vector<float> &field, cv::Mat *img)
{
    initGrids( img );
    calculateDistances( grid1 );
    calculateDistances( grid2 );
    mergeGrids( field );
}

void distanceTransform::initGrids( cv::Mat *img )
{
    cv::Mat binImg = binariseImage( img );

    unsigned char *img_buff = binImg.data;

    for (int y = 0; y < resY; y++)
    {
       for (int x = 0; x < resX; x++)
       {
          if( img_buff[x+y*resX] > 0 )
          {
             assignGridElement( grid1, INSIDE,  x, y );
             assignGridElement( grid2, OUTSIDE, x, y );
          }
          else
          {
             assignGridElement( grid1, OUTSIDE, x, y );
             assignGridElement( grid2, INSIDE,  x, y );
          }
       }
    }

    //cv::imwrite( "binary.png", binImg );
}

void distanceTransform::compareGridPoints(std::vector<Point> &grid, Point &point,
                                          int offsetX, int offsetY, int x, int y)
{
    if(x + offsetX >= 0 && x + offsetX < resX &&
       y + offsetY >= 0 && y + offsetY < resY)
    {
      Point other_point = getGridElement( grid, x+offsetX, y+offsetY );
      other_point.x += offsetX;
      other_point.y += offsetY;

      if( other_point.square_distance() < point.square_distance())
        point = other_point;
    }
}

void distanceTransform::calculateDistances(std::vector<Point> &grid)
{
    // Pass 0: firstly filtering grid in forward manner on both x & y, then on x in reverse manner

    for ( int y1 = 0; y1 < resY; y1++)
    {
        for ( int x1 = 0; x1 < resX; x1++)
        {
           Point point = getGridElement( grid, x1, y1 );
           compareGridPoints( grid, point, -1,  0, x1, y1 );
           compareGridPoints( grid, point,  0, -1, x1, y1 );
           compareGridPoints( grid, point, -1, -1, x1, y1 );
           compareGridPoints( grid, point,  1, -1, x1, y1 );
           assignGridElement( grid, point, x1, y1 );
         }

         for ( int x2 = resX - 1; x2 >= 0; x2--)
         {
            Point point = getGridElement( grid, x2, y1);
            compareGridPoints( grid, point, 1, 0, x2, y1);
            assignGridElement( grid, point, x2, y1);
         }
    }

    // Pass 1: firstly filtering grid in reverse manner on both x & y, then on x in forward manner

    for ( int y3 = resY - 1; y3 >= 0; y3--)
    {
       for ( int x3 = resX - 1; x3 >= 0; x3--)
       {
          Point point = getGridElement( grid, x3, y3);
          compareGridPoints( grid, point,  1,  0, x3, y3);
          compareGridPoints( grid, point,  0,  1, x3, y3);
          compareGridPoints( grid, point, -1,  1, x3, y3);
          compareGridPoints( grid, point,  1,  1, x3, y3);
          assignGridElement( grid, point, x3, y3);
       }

       for ( int x4 = 0; x4 < resX; x4++)
       {
          Point point = getGridElement( grid, x4, y3);
          compareGridPoints( grid, point, -1, 0, x4, y3);
          assignGridElement( grid, point, x4, y3);
       }
    }
}

void distanceTransform::mergeGrids(std::vector<float> &grid)
{
     for( int y = 0; y < resY; y++)
    {
        for( int x = 0; x < resX; x++)
        {
            float dist1 = ( std::sqrt( getGridElement( grid1, x, y ).square_distance()));
            float dist2 = ( std::sqrt( getGridElement( grid2, x, y ).square_distance()));
            grid[x + y * resX] = (dist1 - dist2) * 10.0f / INF;
        }
     }
}

//TODO: check the limits here!
void distanceTransform::smoothDistanceTransformField(std::vector<float> *inField, std::vector<float> *outField)
{
    float *resField = outField->data();
    float *field    = inField->data();
    int res_x = resX,
        res_y = resY;
    int X = resX - bShift,
        Y = resY - bShift;

    float u, v, u_opp, v_opp;
    int x1, y1;
#ifdef USE_OPENMP
#pragma omp parallel shared(resField,field,res_x,res_y,X,Y) private(u,v,u_opp,v_opp,x1,y1)
{
#pragma omp for simd schedule(static, 5)
#endif
    for(int y = 0; y < Y; ++y)
    {
        for(int x = 0; x < X; ++x)
        {
            x1 = clipWithBounds( x, 0, 510 );
            y1 = clipWithBounds( y, 0, 510 );
            u = static_cast<float>(x1) / X;
            v = static_cast<float>(y1) / Y;

            u_opp = 1.0f - u;
            v_opp = 1.0f - v;

            resField[x+y*X] = ( field[ x1 +     y1*res_x] * u_opp + field[(x1+1) +     y1*res_x] * u ) * v_opp +
                              ( field[ x1 + (y1+1)*res_x] * u_opp + field[(x1+1) + (y1+1)*res_x] * u ) * v;
        }
    }
#ifdef USE_OPENMP
}
#endif
}

void distanceTransform::inverseMappingResult(float *res, std::vector<float> SDF, float u_shifted, float v_shifted, cv::Mat affine_m)
{
    cv::invertAffineTransform(affine_m, inv_affine_m);

    float xx = u_shifted * static_cast<float>(resX - bShift);
    float yy = v_shifted * static_cast<float>(resY - bShift);

    cv::Point3d point = cv::Point3d(xx, yy, 1.0);

    float x = std::round((inv_affine_m.at<double>(0, 0) * point.x +
                          inv_affine_m.at<double>(0, 1) * point.y +
                          inv_affine_m.at<double>(0, 2) * point.z)+1.0 );

    float y = std::round((inv_affine_m.at<double>(1, 0) * point.x +
                          inv_affine_m.at<double>(1, 1) * point.y +
                          inv_affine_m.at<double>(1, 2) * point.z)+1.0 );

    float *data = SDF.data();
    float FF1, FF2, aver;
    int res_x = resX - bShift;
    int res_y = resY - bShift;

    //SECTION:inside of the defined picture
    if( (static_cast<int>(x) < res_x && static_cast<int>(y) < res_y) && (x >= 0.0f && y >= 0.0f) )
    {
        *res = getGridElement(data, static_cast<int>(x), static_cast<int>(y), res_x);
    }

    //SECTION: outside of the defined picture, right border
    else if( (static_cast<int>(x) >= res_x && static_cast<int>(y) < res_y) && (x >= 0.0f && y >= 0.0f) )
    {
        FF1  = getGridElement(data, res_x-2, static_cast<int>(y), res_x);
        FF2  = getGridElement(data, res_x-1, static_cast<int>(y), res_x);

        aver = extrapolateVals(FF1, FF2, res_x-2, res_x-1, static_cast<int>(x));
        *res = getGridElement(data, res_x-1, static_cast<int>(y), res_x) + aver;
        if(std::abs(*res) < 2.0f)
          *res *= 2.0f;
    }

    //SECTION: outside of the defined picture, bottom border
    else if( (static_cast<int>(x) < res_x && static_cast<int>(y) >= res_y) && (x >= 0.0f && y >= 0.0f) )
    {
        FF1  = getGridElement(data, static_cast<int>(x), res_y-2, res_x);
        FF2  = getGridElement(data, static_cast<int>(x), res_y-1, res_x);

        aver = extrapolateVals(FF1, FF2, res_y-2, res_y-1, static_cast<int>(y));
        *res = getGridElement(data, static_cast<int>(x), res_y-1, res_x) + aver;
    }

    //SECTION: outside of the defined picture, top border
    else if( static_cast<int>(x) < res_x && (x >= 0.0f && y < 0.0f))
    {
        FF1  = getGridElement(data, static_cast<int>(x), 1, res_x);
        FF2  = getGridElement(data, static_cast<int>(x), 0, res_x);

        aver = extrapolateVals(FF1, FF2, 1, 0, static_cast<int>(y));
        *res = getGridElement(data, static_cast<int>(x), 0, res_x) + aver;
    }

    //SECTION: outside of the defined picture, left border
    else if( static_cast<int>(y) < res_y && (x < 0.0f && y >= 0.0f))
    {
        FF1  = getGridElement(data, 1, static_cast<int>(y), res_x);
        FF2  = getGridElement(data, 0, static_cast<int>(y), res_x);

        aver = extrapolateVals(FF1, FF2, 1, 0, static_cast<int>(x));
        *res = getGridElement(data, 0, static_cast<int>(y), res_x) + aver;
    }

    //SECTION: outside of the defined picture, top right corner
    else if( static_cast<int>(x) >= res_x && (x >= 0.0f && y < 0.0f))
    {
        FF1  = getGridElement(data, res_x-2, 1, res_x);
        FF2  = getGridElement(data, res_x-1, 0, res_x);

        aver = extrapolateVals(FF1, FF2, 1, 0, static_cast<int>(x));
        *res = getGridElement(data, res_x-1, 0, res_x) + aver;
    }

    //SECTION: outside of the defined picture, bottom right corner
    else if( static_cast<int>(x) >= res_x && static_cast<int>(y) >= res_y)
    {
        FF1  = getGridElement(data, res_x-2, res_y-2, res_x);
        FF2  = getGridElement(data, res_x-1, res_y-1, res_x);

        aver = extrapolateVals(FF1, FF2, res_y-2, res_y-1, static_cast<int>(x));
        *res = getGridElement(data, res_x-1, res_y-1, res_x) + aver;
    }

    //SECTION: outside of the defined picture, bottom left corner
    else if ( static_cast<int>(y) >= res_y && (x < 0.0f && y >= 0.0f))
    {
        FF1  = getGridElement(data, 1, res_y-2, res_x);
        FF2  = getGridElement(data, 0, res_y-1, res_x);

        aver = extrapolateVals(FF1, FF2, res_y-2, res_y-1, static_cast<int>(x));
        *res = getGridElement(data, 0, res_y-1, res_x) + aver;
    }

    //SECTION: outside of the defined picture, top left corner
    else if(x < 0.0f && y < 0.0f)
    {
        FF1  = getGridElement(data, 1, 1, res_x);
        FF2  = getGridElement(data, 0, 0, res_x);

        aver = extrapolateVals(FF1, FF2, res_x-2, res_x-1, static_cast<int>(x));
        *res = getGridElement(data, res_x-1, res_y-1, res_x) + aver;
    }
    else
    {
        std::cerr << "No conditions in inverse SDF procedure are satisfiied! abort!" << std::endl;
    }
}

}//namespace metamorphosis
