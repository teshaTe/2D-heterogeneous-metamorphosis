#ifndef H_COLOUR_INTERPOALTION_CLASS
#define H_COLOUR_INTERPOALTION_CLASS

#include "include/colourSegmentation.h"

#include <memory>

#include <glm/glm.hpp>

#ifdef USE_OPENMP
    #include <omp.h>
#endif

namespace metamorphosis {

enum class DrawMode
{
  WHITE,
  BLOCK_COLOR,
  SIMPLE_BLEND,
  SIMPLE_CLOSEST
};

enum class ColorMode
{
  RGB,
  HSV,
  CIELab
};

struct basicColors
{
    glm::vec3 green   = {};
    glm::vec3 red     = {};
    glm::vec3 blue    = {};
    glm::vec3 orange  = {};
    glm::vec3 purple  = {};
    glm::vec3 yellow  = {};
    glm::vec3 magenta = {};
};

class colourInter
{
public:
    colourInter(cv::Mat initIm, cv::Mat targIm, std::vector<float> f1, std::vector<float> f2,
                int width, int height, DrawMode dMode, ColorMode cMode , int colSegmNum);
    ~colourInter() { }

    glm::vec4 computeSTTI_noCorrespondences( float f1, float f2, float unSmoothF1, float unSmoothF2, glm::vec2 uv );
    glm::vec4 computeSTTI_withCorrespondences( float f1, float f2, float unSmoothF1, float unSmoothF2, glm::vec2 uv );

    glm::vec4 getPixelColor( unsigned char *pixBuf, size_t step, int numChannels, glm::vec2 uv );

    void convertToRGB   ( float *r, float *g, float *b );
    void convertFromRGB ( float *r, float *g, float *b );
    void clampColor( glm::vec4 *col );

private:

    void constructCorrespondences(cv::Mat img1 , cv::Mat img2);

    void useBlendingColors(glm::vec4 *col, cv::Mat img, std::vector<float> *fun, float funVal , glm::vec2 uv);
    void useClosestColors(glm::vec4 *col, cv::Mat img, std::vector<float> *fun, glm::vec2 uv );

    void convertRGBtoHSV   ( float *r, float *g, float *b );
    void convertRGBtoCIELab( float *r, float *g, float *b );

    void convertHSVtoRGB   ( float *r, float *g, float *b );
    void convertCIELabToRGB( float *r, float *g, float *b );

    inline int clipWithBounds( int n, int n_min, int n_max ) { return n > n_max ? n_max :( n < n_min ? n_min : n ); }
    inline void convertToUV( float *x, float *y )         { *x /= resX; *y /=resY; }
    inline float min3(float a, float b, float c)          { return std::min(std::min(a, b), c); }
    inline float max3(float a, float b, float c)          { return std::max(std::max(a, b), c); }
    inline float min4(float a, float b, float c, float d) { return std::min(a, std::min(b, std::min(c,d))); }
    inline float max4(float a, float b, float c, float d) { return std::max(a, std::max(b, std::max(c,d))); }

private:
    DrawMode curDrawMode;
    ColorMode curColMode;
    int resX, resY;

    cv::Mat inImg, targImg;
    unsigned char *initImBuf, *targImBuf;

    std::vector<float> F1, F2;

    //variables for color segmentation only
    std::shared_ptr<colourSegm> colSegm;
    std::vector<cv::Vec3b> im1Colors, im2Colors;
    cv::Mat classificator1, classificator2;
    colNode *root1, *root2;
};

} //namespace metamorphosis
#endif //H_COLOUR_INTERPOALTION_CLASS
