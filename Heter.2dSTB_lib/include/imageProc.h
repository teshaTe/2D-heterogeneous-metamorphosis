#ifndef H_IMAGE_PROCESSING_CLASS
#define H_IMAGE_PROCESSING_CLASS

/*
 * This class provides basic opencv-based tools:
 * 1. for defining circumscribed circles/rectangles and finding the maximum one;
 * 2. for computing steps for the Affine translation operation
 */
#include <opencv4/opencv2/imgproc.hpp>
#include <glm/glm.hpp>
#include <vector>

namespace hstb {

class imageProc
{
public:
    imageProc(const cv::Mat &img1, const cv::Mat &img2);
    imageProc() { thresh = 100; maxThresh = 255; }
    ~imageProc() {}

    //Functions for obtaining circumscribed circles/rectangles around input 2d objects
    void getMaxCircumscribedCircles(float *r1, float *r2, glm::vec2 *cen1, glm::vec2 *cen2);
    void getMaxCircumscribedRectangles(cv::Rect *maxRec1, cv::Rect *maxRec2, glm::vec2 *cen1, glm::vec2 *cen2);

    void setNewImages(const cv::Mat &img1, const cv::Mat &img2);
    //Support functions for handling affine translation of the object
    void getTransformSteps(float *stX, float *stY, float shiftX, float shiftY, const int totalFrames);
    glm::vec2 getTransformSteps();
    cv::Mat getAffineTransformedImage(cv::Mat *img, float stX, float stY );

    inline cv::Mat getAffineMatrixTranform() { return (cv::Mat_<double>(2 ,3) << 1, 0, getTransformSteps().x/image1.cols,
                                                                                 0, 1, getTransformSteps().y/image1.rows); }
private:
    void createCircumCircle(int, void*);
    void createCircumRectangle(int, void*);
    cv::Mat makeGrayImage(cv::Mat *img);

private:
    cv::Mat image1, image2;
    cv::Mat grayImg1, grayImg2;
    cv::Mat threshOutput1, threshOutput2;

    int thresh, maxThresh;
    float totalStX, totalStY;

    std::vector<cv::Rect> boundRect1, boundRect2;
    std::vector<cv::Point2f> center1, center2;
    std::vector<float> radius1, radius2;
    std::vector<float> image2Shifted;
};

} //namespace hstb
#endif //#ifndef H_IMAGE_PROCESSING_CLASS
