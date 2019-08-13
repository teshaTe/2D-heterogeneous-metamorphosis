#ifndef H_IMAGE_SEGMENTATION_CLASS
#define H_IMAGE_SEGMENTATION_CLASS

#include <opencv4/opencv2/opencv.hpp>

namespace metamorphosis {

class imgSegm
{
public:
    imgSegm();
    ~imgSegm() {}

    void computeWatershedSegm( cv::Mat img );
    cv::Mat getColoredSegmImage();

    inline void updateBinarizationThres( double binThr1, double binThr2 ) { binThres1  = binThr1;   binThres2 = binThr2; }
    inline void updatePeakThres( double peakThr1, double peakThr2 )       { peakThres1 = peakThr1; peakThres2 = peakThr2; }
    inline std::vector<std::vector<cv::Point>> getContours() { return contours; }
    inline cv::Mat getMarkers() { return markers * 10000; }
    inline cv::Mat getDDTImg() { return distTr; }

private:
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat markers, mark, distTr;
    double binThres1,  binThres2;
    double peakThres1, peakThres2;
};

}// namespace metamorphosis
#endif // define H_IMAGE_SEGMENTATION_CLASS
