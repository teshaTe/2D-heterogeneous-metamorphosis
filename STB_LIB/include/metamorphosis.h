#ifndef H_METAMORPHOSIS_CLASS
#define H_METAMORPHOSIS_CLASS

#include "distanceTransform2D.h"
#include "colourInterpolation.h"
#include "imageProc.h"

#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <memory>

namespace metamorphosis {

class metamorphosis2D
{
public:
    metamorphosis2D( cv::Mat inputImg, cv::Mat targImg, ColorMode colorMode, DrawMode drawMode,
                     int totalFrames, float scale, bool autoMode = true );
    ~metamorphosis2D() { }

    void calculateMetamorphosis();

    inline void setManualSTBcoefficients( float A0, float A1, float A2, float A3 ) { a0 = A0; a1 = A1; a2 = A2; a3 = A3; }
    inline void setBackgroundImage( std::string path ) { background = cv::imread( path, cv::IMREAD_UNCHANGED ); }
    inline void setOutputPath( std::string path ) { outputPath = path; }

private:
    void useImageBackground(cv::Mat *dst, cv::Mat curImg);
    void scaleImage(cv::Mat *dst, cv::Mat src, float scale , int cv_interpolation);

    void setAutoSTBcoefficients();
    void calcA1A2coefficient();
    void calcA0coefficient();
    void caclA3Coefficient();

    void calcAffineTransformedDT(float shiftX, float shiftY);
    bool check2CircleCrosssection();

    float smoothHalfCylinder1( float functionVal, float time );
    float smoothHalfCylinder2( float functionVal, float time );
    float obtainF3Values( glm::vec2 uv, float time );

    glm::vec4 obtainResultingColor(glm::vec2 coord, int frameNumber );
    glm::vec4 obtainBlendingResult( glm::vec2 uv, float time );

    glm::vec4 getResultingColor( glm::vec2 uv, float f1, float f2, float f1BeforeSm, float f2BeforeSm );

    inline int clipWithBounds( int n, int n_min, int n_max ) { return n > n_max ? n_max :( n < n_min ? n_min : n ); }
    inline void convertToUV( float *x, float *y )     { *x /= resX; *y /=resY; }

    inline float min4(float a, float b, float c, float d) { return std::min(a, std::min(b, std::min(c,d))); }
    inline float max4(float a, float b, float c, float d) { return std::max(a, std::max(b, std::max(c,d))); }

    inline float intersectFunctions(float a, float b) { return a + b - std::sqrt(a * a + b * b); }
    inline float unionFunctions(float a, float b)     { return a + b + std::sqrt(a * a + b * b); }
    inline float subtractFunctions(float a, float b)  { return intersectFunctions(a, -b); }

private:
    DrawMode curDrawMode;
    ColorMode curColorMode;

    std::string outputPath;

    cv::Mat inputShape, targetShape, background, resultImg, affine_mat;

    unsigned char *targImBuf, *inputImBuf, *resultImBuf;
    size_t stepTargIm, stepInputImg, stepResImg;
    int inputChs, targChs, resChs;
    float finScale;

    int curFrame, frameCount, resX, resY;

    float a0, a1, a2, a3;
    float b0_1, b1_1, b2_1, b3_1;
    float b0_2, b1_2, b2_2, b3_2;

    float shiftX, shiftY;
    float minDist;

    float rad1, rad2;
    glm::vec2 cen1, cen2;

    cv::Rect rec1, rec2;
    glm::vec2 rec1_c, rec2_c;

    std::vector<float> F1, F2, F2_m;
    std::vector<float> time;

    std::shared_ptr<distanceTransform> DT;
    std::shared_ptr<colourInter> colInter;
    std::shared_ptr<imageProc> imgProc1, imgProc2;

};

} //namespace metamorphosis
#endif //#define H_METAMORPHOSIS_CLASS
