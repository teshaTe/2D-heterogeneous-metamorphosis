#include "include/imageProc.h"

#ifdef USE_DEBUG_INFO
    #include <opencv2/imgcodecs.hpp>
    #include <opencv4/opencv2/highgui.hpp>
#endif

namespace metamorphosis {

imageProc::imageProc(cv::Mat *img1, cv::Mat *img2)
{
    img1->copyTo(image1);
    img2->copyTo(image2);

    grayImg1 = makeGrayImage( &image1 );
    grayImg2 = makeGrayImage( &image2 );

    thresh    = 70;
    maxThresh = 255;

#if defined(USE_CONE_F3)
    createCircumCircle( 0, 0 );
#endif

#ifdef USE_PYRAMID_F3
    createRectangle( 0, 0 );
#endif
}

imageProc::imageProc(cv::Mat *img1, cv::Mat *img2, bool useRectangle)
{
    img1->copyTo(image1);
    img2->copyTo(image2);

    grayImg1 = makeGrayImage( &image1 );
    grayImg2 = makeGrayImage( &image2 );

    thresh    = 70;
    maxThresh = 255;

    if( useRectangle )
        createRectangle( 0, 0 );
    else
        createCircumCircle( 0, 0 );
}

cv::Mat imageProc::makeGrayImage(cv::Mat *img)
{
    cv::Mat srcGray;
    cv::cvtColor( *img, srcGray, cv::COLOR_BGR2GRAY);
    cv::blur( srcGray, srcGray, cv::Size(3,3) );
    return srcGray;
}

void imageProc::createCircumCircle(int, void *)
{
    std::vector<cv::Vec4i> hierarchy1, hierarchy2;

    /// Detect edges using Threshold
    cv::threshold( grayImg1, threshOutput1, thresh, 255, cv::THRESH_BINARY );
    cv::threshold( grayImg2, threshOutput2, thresh, 255, cv::THRESH_BINARY );

    /// Find contours
    cv::findContours( threshOutput1, contours1, hierarchy1,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point2f(0, 0) );
    cv::findContours( threshOutput2, contours2, hierarchy2,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point2f(0, 0) );

    /// Approximate contours to polygons + get bounding rects and circles
    /// First Picture
    contoursPoly1.resize(contours1.size());
    center1.resize( contours1.size() );
    radius1.resize( contours1.size() );

    ///Second Picture
    contoursPoly2.resize( contours2.size() );
    center2.resize( contours2.size() );
    radius2.resize( contours2.size() );

    for(size_t i = 0; i < contours1.size(); i++ )
    {
       cv::approxPolyDP( cv::Mat(contours1[i]), contoursPoly1[i], 3, true );
       cv::minEnclosingCircle( (cv::Mat)contoursPoly1[i], center1[i], radius1[i] );
    }

    for(size_t j = 0; j < contours2.size(); j++ )
    {
        cv::approxPolyDP( cv::Mat(contours2[j]), contoursPoly2[j], 3, true );
        cv::minEnclosingCircle( (cv::Mat)contoursPoly2[j], center2[j], radius2[j] );
    }

    #ifdef USE_DEBUG_INFO
        cv::RNG rng(12345);
        cv::Mat drawing1 = cv::Mat::zeros(image1.rows, image1.cols, image1.type());
        cv::Mat drawing2 = cv::Mat::zeros(image2.rows, image2.cols, image2.type());

        for( int i = 0; i < contours1.size(); i++ )
        {
           cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
           cv::drawContours( drawing1, contoursPoly1, i, color, 2, 8, hierarchy1, 0, cv::Point() );
           cv::circle( drawing1, center1[i], (int)radius1[i], color, 2, 8, 0);
        }

        for( int i = 0; i< contours2.size(); i++ )
        {
           cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
           cv::drawContours( drawing2, contoursPoly2, i, color, 2, 8, hierarchy2, 0, cv::Point() );
           cv::circle( drawing2, center2[i], (int)radius2[i], color, 2, 8, 0);
        }

        cv::Mat blended_dst = cv::Mat::zeros(image2.rows, image2.cols, image2.type());
        cv::addWeighted( drawing1, 0.5, drawing2, 0.5, 0.0, blended_dst);

        cv::imwrite("blended_image.jpg", blended_dst);
        cv::imwrite("image1.jpg", drawing1);
        cv::imwrite("image2.jpg", drawing2);
#endif
}

void imageProc::createRectangle(int, void *)
{
    std::vector<cv::Vec4i> hierarchy1, hierarchy2;

    /// Detect edges using Threshold
    cv::threshold( grayImg1, threshOutput1, thresh, 255, cv::THRESH_BINARY );
    cv::threshold( grayImg2, threshOutput2, thresh, 255, cv::THRESH_BINARY );

    /// Find contours
    cv::findContours( threshOutput1, contours1, hierarchy1,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point2f(0, 0) );
    cv::findContours( threshOutput2, contours2, hierarchy2,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point2f(0, 0) );

    /// Approximate contours to polygons + get bounding rects and circles
    /// First Picture
    contoursPoly1.resize(contours1.size());
    boundRect1.resize( contours1.size() );

    ///Second Picture
    contoursPoly2.resize(contours2.size());
    boundRect2.resize( contours2.size() );

    for(int i = 0; i < contours1.size(); i++ )
    {
       cv::approxPolyDP( cv::Mat(contours1[i]), contoursPoly1[i], 3, true );
       boundRect1[i] = cv::boundingRect( cv::Mat(contoursPoly1[i]) );
    }

    for(int j = 0; j < contours2.size(); j++ )
    {
        cv::approxPolyDP( cv::Mat(contours2[j]), contoursPoly2[j], 3, true );
        boundRect2[j] = cv::boundingRect( cv::Mat(contoursPoly2[j]) );
    }

#ifdef USE_DEBUG_INFO
    cv::RNG rng(12345);
    cv::Mat drawing1 = cv::Mat::zeros(image1.rows, image1.cols, image1.type());
    cv::Mat drawing2 = cv::Mat::zeros(image2.rows, image2.cols, image2.type());

    for( int i = 0; i< contours1.size(); i++ )
    {
       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       cv::drawContours( drawing1, contoursPoly1, i, color, 2, 8, hierarchy1, 0, cv::Point() );
       cv::rectangle( drawing1, boundRect1[i], color, 2, 8, 0 );
    }

    for( int i = 0; i< contours2.size(); i++ )
    {
       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       cv::drawContours( drawing2, contoursPoly2, i, color, 2, 8, hierarchy2, 0, cv::Point() );
       cv::rectangle( drawing1, boundRect2[i], color, 2, 8, 0 );
    }

    cv::Mat blended_dst = cv::Mat::zeros(image2.rows, image2.cols, image2.type());
    cv::addWeighted( drawing1, 0.5, drawing2, 0.5, 0.0, blended_dst);

    cv::imwrite("blended_image.jpg", blended_dst);
    cv::imwrite("image1.jpg", drawing1);
    cv::imwrite("image2.jpg", drawing2);
#endif
}

void imageProc::getMaxCircles(float *r1, float *r2, glm::vec2 *cen1, glm::vec2 *cen2)
{
    *r1 = *std::max_element( radius1.begin(), radius1.end() );
    *r2 = *std::max_element( radius2.begin(), radius2.end() );

    int i1 = std::distance( radius1.begin(), std::find(radius1.begin(), radius1.end(), *r1 ));
    *cen1  = glm::vec2(center1[i1].x / image1.cols, center1[i1].y / image1.rows);

    int i2 = std::distance( radius2.begin(), std::find(radius2.begin(), radius2.end(), *r2 ));
    *cen2  = glm::vec2(center2[i2].x / image2.cols, center2[i2].y / image2.rows);
}

void imageProc::getMaxRectangles(cv::Rect *maxRec1, cv::Rect *maxRec2 )
{
    int max_s1 = boundRect1[0].width * boundRect1[0].height;
    int max_s2 = boundRect2[0].width * boundRect2[0].height;
    size_t index = 0;

    for ( size_t i = 0; i < boundRect1.size(); i++ )
    {
       if(max_s1  < boundRect1[i].width * boundRect1[i].height ||
          max_s1 == boundRect1[i].width * boundRect1[i].height)
       {
          max_s1 = boundRect1[i].width * boundRect1[i].height;
          index = i;
       }
    }

    *maxRec1 = boundRect1[index];
    index = 0;

    for ( size_t i = 0; i < boundRect2.size(); i++ )
    {
       if(max_s2  < boundRect2[i].width * boundRect2[i].height ||
          max_s2 == boundRect2[i].width * boundRect2[i].height)
       {
          max_s2 = boundRect2[i].width * boundRect2[i].height;
          index = i;
       }
    }

    *maxRec2 = boundRect2[index];
}

void imageProc::getRectangleCenters(glm::vec2 *cen1, glm::vec2 *cen2)
{
    cv::Rect rec1, rec2;
    getMaxRectangles( &rec1, &rec2 );

    *cen1 = glm::vec2( rec1.x + rec1.width / 2.0f, rec1.y + rec1.height / 2.0f );
    *cen2 = glm::vec2( rec2.x + rec2.width / 2.0f, rec2.y + rec2.height / 2.0f );
}

void imageProc::getShiftingStepXY(float *stepX, float *stepY, float shiftX, float shiftY, int totalFrames)
{
    if(shiftX != 0.0f && shiftY != 0.0f)
    {
        *stepX = std::floor(shiftX / totalFrames);
        *stepY = std::floor(shiftY / totalFrames);
        totalStX += std::abs(*stepX);
        totalStY += std::abs(*stepY);
    }
    else if(shiftX != 0.0f && shiftY == 0.0f)
    {
        *stepX = std::floor(shiftX / totalFrames);
        *stepY = 0.0f;
        totalStX += std::abs(*stepX);
    }
    else if(shiftX == 0.0f && shiftY != 0.0f)
    {
        *stepX = 0.0f;
        *stepY = std::floor(shiftY / totalFrames);
        totalStY += std::abs(*stepY);
    }

    if(totalStX >= std::abs(shiftX) || totalStY >= std::abs(shiftY))
    {
        *stepX = 0.0f;
        *stepY = 0.0f;
    }
}

glm::vec2 imageProc::getUVshiftImage2()
{
    glm::vec2 cen1, cen2;
    getRectangleCenters( &cen1, &cen2 );

    glm::vec2 result = glm::vec2( cen1.x - cen2.x, cen1.y - cen2.y );
    return result;
}

cv::Mat imageProc::getAffineTransformedImage(cv::Mat *img, cv::Mat *affine_m, float stepX, float stepY)
{
    cv::Mat dst = cv::Mat( img->size(), img->type());
    *affine_m = (cv::Mat_<double>(2 ,3) << 1, 0, stepX,
                                           0, 1, stepY);
    cv::warpAffine( *img, dst, *affine_m, img->size(), cv::WARP_FILL_OUTLIERS);

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

}//namespace metamorphosis
