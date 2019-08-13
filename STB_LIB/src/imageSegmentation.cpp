#include "include/imageSegmentation.h"

namespace metamorphosis {

imgSegm::imgSegm()
{
    binThres1 = 40.0;
    binThres2 = 255.0;

    peakThres1 = 0.4;
    peakThres2 = 1.0;
}

void imgSegm::computeWatershedSegm(cv::Mat img)
{
    //crearing a kernel for accuting/sharpening image;
    //kernel defined as an approximation of second derivative
    cv::Mat kernelLap = ( cv::Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

    //performing laplacian filtering of the image
    cv::Mat imgLaplacian, imgSharp;
    imgSharp = img;

    cv::filter2D( imgSharp, imgLaplacian, CV_32F, kernelLap );
    img.convertTo( imgSharp, CV_32F );
    cv::Mat imgResult = imgSharp - imgLaplacian;

    //converig the resulting image back to 8bit gray-scale
    imgResult.convertTo( imgResult, CV_8UC3 );
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3 );

    //binarising image
    cv::Mat binImg;
    cv::cvtColor( imgResult, binImg, cv::COLOR_BGR2GRAY );
    cv::threshold( binImg, binImg, binThres1, binThres2, cv::THRESH_BINARY | cv::THRESH_OTSU );

    //obtaining distances for the image using distance transform;
    //normalising the resulting distances to [0,1]
    cv::distanceTransform( binImg, distTr, cv::DIST_L2, 3 );
    cv::normalize( distTr, distTr, 0, 1, cv::NORM_MINMAX );

    //computing markers for the foreground objects
    cv::threshold( distTr, distTr, peakThres1, peakThres2, cv::THRESH_BINARY );

    //dilating the image
    cv::Mat kernelDil = cv::Mat::ones( 3, 3, CV_8UC1 );
    cv::dilate( distTr, distTr, kernelDil );

    cv::Mat distTr_to_8U;
    distTr.convertTo( distTr_to_8U, CV_8U );

    //find total markers
    cv::findContours( distTr_to_8U, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
    markers = cv::Mat::zeros( distTr.size(), CV_32SC1 );

    for ( size_t i = 0 ; i < contours.size(); i++)
    {
        cv::drawContours( markers, contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i)+1), -1 );
        cv::circle( markers, cv::Point( 5, 5 ), 3, CV_RGB(255, 255, 255), -1 );
    }


    //performing the watershed algorithm
    cv::watershed( imgResult, markers );

    mark = cv::Mat::zeros( markers.size(), CV_8UC1 );
    markers.convertTo( mark, CV_8UC1 );
    cv::bitwise_not( mark, mark );
    cv::imshow( "markers", mark );
    cv::waitKey(0);
}

cv::Mat imgSegm::getColoredSegmImage()
{
    std::vector<cv::Vec3b> colors;
    for ( size_t i = 0 ; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform( 0, 255 );
        int g = cv::theRNG().uniform( 0, 255 );
        int r = cv::theRNG().uniform( 0, 255 );
        colors.push_back( cv::Vec3b( (uchar)b, (uchar)g, (uchar)r ));
    }

    cv::Mat dst = cv::Mat::zeros( markers.size(), CV_8UC3 );
    unsigned char *markersBuff = markers.data;
    unsigned char *dstBuff = dst.data;

    for( int y = 0; y < markers.rows; y++ )
    {
        for (int x = 0; x < markers.cols; x++)
        {
            int index = markersBuff[x*markers.channels()+y*markers.step];
            if( index > 0 && index <= static_cast<int>( contours.size()) )
            {
                dstBuff[x*dst.channels()     + y*dst.step] = colors[index - 1][0];
                dstBuff[x*dst.channels() + 1 + y*dst.step] = colors[index - 1][1];
                dstBuff[x*dst.channels() + 2 + y*dst.step] = colors[index - 1][2];
            }
            else {
                dstBuff[x*dst.channels()     + y*dst.step] = 0;
                dstBuff[x*dst.channels() + 1 + y*dst.step] = 0;
                dstBuff[x*dst.channels() + 2 + y*dst.step] = 0;
            }
        }
    }

    return dst;
}


} //namespace metamorphosis
