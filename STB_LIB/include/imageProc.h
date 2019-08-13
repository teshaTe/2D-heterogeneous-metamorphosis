#ifndef H_IMAGE_PROCESSING_CLASS
#define H_IMAGE_PROCESSING_CLASS

#include <opencv4/opencv2/imgproc.hpp>
#include <glm/glm.hpp>

#include <vector>

namespace metamorphosis {

class imageProc
{
public:
    imageProc( cv::Mat *img1, cv::Mat *img2 );
    imageProc( cv::Mat *img1, cv::Mat *img2, bool useRectangle );
    imageProc() { thresh = 70; maxThresh = 255; }
    ~imageProc() { }

    void getMaxCircles( float *r1, float *r2, glm::vec2 *cen1, glm::vec2 *cen2 );
    void getMaxRectangles(cv::Rect *maxRec1 , cv::Rect *maxRec2);
    void getRectangleCenters( glm::vec2 *cen1, glm::vec2 *cen2 );

    //affine translation
    void getShiftingStepXY( float *stepX, float *stepY, float shiftX, float shiftY, int totalFrames );
    glm::vec2 getUVshiftImage2();
    cv::Mat getAffineTransformedImage(cv::Mat *img, cv::Mat *affine_m, float stepX, float stepY );

    inline cv::Mat getAffineMatrixTranslation() { return (cv::Mat_<double>(2 ,3) << 1, 0, getUVshiftImage2().x,
                                                                                    0, 1, getUVshiftImage2().y); }
private:
    void createCircumCircle( int, void* );
    void createRectangle( int, void* );
    cv::Mat makeGrayImage( cv::Mat *img );

private:
    cv::Mat image1, image2;
    cv::Mat grayImg1, grayImg2;
    cv::Mat threshOutput1, threshOutput2;

    int thresh, maxThresh;
    float totalStX, totalStY;

    std::vector<std::vector<cv::Point> > contoursPoly1, contoursPoly2;
    std::vector<std::vector<cv::Point> > contours1, contours2;
    std::vector<cv::Rect> boundRect1, boundRect2;
    std::vector<cv::Point2f> center1, center2;
    std::vector<float> radius1, radius2;
    std::vector<float> image2Shifted;
};

} // namespace metamorphosis
#endif // H_IMAGE_PROCESSING_CLASS
