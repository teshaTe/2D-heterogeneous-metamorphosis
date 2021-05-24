#ifndef H_HETER_STB_CLASS
#define H_HETER_STB_CLASS

/*
 * This class relies on core Space-Time Blending paper:
 * @article{PASKO:2004,
            author  = {G. I. {Pasko} and A. A. {Pasko} and T. L. {Kunii}},
            journal = {IEEE Computer Graphics and Applications},
            title   = {Bounded blending for function-based shape modeling},
            year    = {2005},
            volume  = {25},
            number  = {2},
            pages   = {36-45},
            doi     = {https://doi.ieeecomputersociety.org/10.1109/MCG.2005.37}
            }

    and introduced methods in the current paper:
    1) basic algorithm, subsection 4.1, p. 10-11;
    2) enhancements od the basic algorithm, section 5 intro p. 12 -13;
    3) methods themselves, section 5 p. 13 - 21;
*/


#include "include/distanceTransform2D.h"
#include "include/STTI.h"
#include "include/imageProc.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <glm/glm.hpp>

#include <memory>
#include <vector>
#include <string>

namespace hstb {

enum class BoundingSolid
{
    none,
    truncCone,
    truncPyramid,
    halfPlanes
};

class heter2dSTB
{
public:
    heter2dSTB(const cv::Mat &inImg, const cv::Mat &targImg, ColorMode colorMode,
                                DrawMode drawMode, int totalFrames, float scale);
    ~heter2dSTB() = default;

    void computeMetamorphosis(BoundingSolid f3, bool useAffineTr, bool autoMode = true);
    void computeCrossDissolve();

    inline void setManualSTBcoefficients(float A0, float A1, float A2, float A3 ) { a0 = A0; a1 = A1; a2 = A2; a3 = A3; }
    inline void setBackgroundImage(std::string path) { background = cv::imread(path, cv::IMREAD_UNCHANGED); }
    inline void setOutputPath(std::string path)      { outputPath = path; }

private:
    void useImageBackground(cv::Mat *dst, const cv::Mat &curImg);
    void scaleImage(cv::Mat *dst, const cv::Mat &src, float scale, int cv_interpolation);

    void setAutoSTBcoefficients();
    void calcA1A2coefficients();
    void calcA1A2coefficients_test();

    void calcA0coefficient();
    void caclA3Coefficient();
    bool check2CircleCrosssection();

    void calcAffineTransformedDT(float shiftX, float shiftY);

    float smoothHalfCylinder1(float functionVal, float time);
    float smoothHalfCylinder2(float functionVal, float time);
    float obtainF3Values(glm::vec2 uv, float time);

    glm::vec4 computePixelColour(glm::vec2 coord, int frameNumber );
    glm::vec4 obtainBlendingResult( glm::vec2 uv, float time );

    inline void setBoundingSolid(BoundingSolid solid) { bSolid = solid; }
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
    BoundingSolid bSolid;
    bool useAffineTrans;

    std::string outputPath;

    cv::Mat inShape, targShape, background, resultImg, affine_mat;

    unsigned char *targImBuf, *inputImBuf, *resultImBuf;
    size_t stepTargIm, stepInImg, stepResImg;
    int inChs, targChs, resChs;
    float finScale;

    int curFrame, frameCount, resX, resY;

    float a0, a1, a2, a3;
    float b0_1, b1_1, b2_1, b3_1;
    float b0_2, b1_2, b2_2, b3_2;

    glm::vec2 stepXY;
    float minDist;

    float rad1, rad2;
    glm::vec2 cen1, cen2;

    cv::Rect rec1, rec2;
    glm::vec2 rec1_c, rec2_c;

    std::vector<float> F1, F2;
    std::vector<float> time;

    std::shared_ptr<SEDTtransform> DT;
    std::shared_ptr<STTImethod> STTI;
    std::shared_ptr<imageProc> imgProc;
};

} //namespace hstb
#endif //H_HETER_STB_CLASS
