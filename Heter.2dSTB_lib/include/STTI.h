#ifndef H_STTI_CLASS
#define H_STTI_CLASS

/*
 * This class relies on core Space-Time Transfinite Interpoaltion paper:
 *      @ARTICLE{SANCHEZ:2015,
                 author  = {M. Sanchez and O. Fryazinov and V. Adzhiev and P. Comninos and A. Pasko},
                 title   = {Space-Time Transfinite Interpolation of Volumetric Material Properties},
                 journal = {IEEE Transactions on Visualization and Computer Graphics},
                 volume  = {21},
                 number  = {2},
                 pages   = {278-288},
                 month   = {feb},
                 year    = {2015},
                 doi     = {https://doi.org/10.1109/TVCG.2014.2356196}
                 }
    This class also provides three pairs of colour coversions between RGB, HSV and CIELab
 */

#include <opencv2/core.hpp>
#include <glm/glm.hpp>
#include <vector>

namespace hstb {

enum class DrawMode
{
    NO_COLOR,
    COLOR_BLOCK,
    COLOR_BLEND,
    COLOR_CLOSEST
};

enum class ColorMode
{
    RGB,
    HSV,
    CIELab
};

class STTImethod
{
public:
    STTImethod(const cv::Mat &inImg, const cv::Mat &targImg, int res_x,
                            int res_y, ColorMode cMode, DrawMode dMode);
    ~STTImethod() = default;

    glm::vec4 computeSTTI(std::vector<float> &F1, std::vector<float> &F2, float f1,
                          float f2, float unSmoothF1, float unSmoothF2, glm::vec2 uv);
    glm::vec4 getPixelColor(const unsigned char *pixBuf, const size_t step, const int numChannels, glm::vec2 uv);

    glm::vec4 convertToRGB(glm::vec4 col);
    glm::vec4 convertFromRGB(glm::vec4 col);
    glm::vec4 clampColor(glm::vec4 col);

    inline void setNewImages(const cv::Mat &img1, const cv::Mat &img2) { in_img = img1; initImBuf = img1.data;
                                                                         targ_img = img2; targImBuf = img2.data; }

private:
    glm::vec4 useBlendingColors(const cv::Mat &img, glm::vec4 colP, std::vector<float> &fun, float funVal, glm::vec2 uv);
    glm::vec4 useClosestColors(const cv::Mat &img, std::vector<float> &fun, glm::vec2 uv);

    glm::vec4 convertRGBtoHSV(glm::vec4 col);
    glm::vec4 convertHSVtoRGB(glm::vec4 col);
    glm::vec4 convertRGBtoCIELab(glm::vec4 col);
    glm::vec4 convertCIELabtoRGB(glm::vec4 col);

    inline int clipWithBounds(int n, int n_min, int n_max) { return n > n_max ? n_max :( n < n_min ? n_min : n ); }
    inline void convertToUV(float *x, float *y)            { *x /= resX; *y /= resY; }
    inline float min3(float a, float b, float c)           { return std::min(std::min(a, b), c); }
    inline float max3(float a, float b, float c)           { return std::max(std::max(a, b), c); }
    inline float min4(float a, float b, float c, float d)  { return std::min(a, std::min(b, std::min(c,d))); }
    inline float max4(float a, float b, float c, float d)  { return std::max(a, std::max(b, std::max(c,d))); }

private:
    DrawMode curDrawMode;
    ColorMode curColMode;
    int resX, resY;

    cv::Mat in_img, targ_img;
    unsigned char *initImBuf, *targImBuf;
};

} //namespace hstb
#endif //H_STTI_CLASS
