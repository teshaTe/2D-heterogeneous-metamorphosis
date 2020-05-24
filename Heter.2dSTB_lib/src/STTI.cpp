#include "include/STTI.h"

#include <iostream>

#ifdef USE_OPENMP
    #include <omp.h>
#endif
namespace hstb {

STTImethod::STTImethod(const cv::Mat &inImg, const cv::Mat &targImg, int res_x, int res_y, ColorMode cMode, DrawMode dMode)
{
    in_img      = inImg;
    targ_img    = targImg;
    initImBuf   = inImg.data;
    targImBuf   = targImg.data;
    curDrawMode = dMode;
    curColMode  = cMode;
    resX = res_x;
    resY = res_y;
}

glm::vec4 STTImethod::computeSTTI(std::vector<float> &F1, std::vector<float> &F2, float f1,
                                  float f2, float unSmoothF1, float unSmoothF2, glm::vec2 uv)
{
    glm::vec4 objCol(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 zero(0.0, 0.0, 0.0, 1.0);

    glm::vec4 col1 = getPixelColor(initImBuf, in_img.step,   in_img.channels(),   uv);
    glm::vec4 col2 = getPixelColor(targImBuf, targ_img.step, targ_img.channels(), uv);

    if(col1 == zero || col2 == zero)
    {
        switch (curDrawMode)
        {
            case DrawMode::NO_COLOR:
                if(col1 == zero) col1 = glm::vec4(1.0, 1.0, 1.0, 1.0);
                if(col2 == zero) col2 = glm::vec4(1.0, 1.0, 1.0, 1.0);
            break;
            case DrawMode::COLOR_BLEND:
                col1 = useBlendingColors(in_img,   col1, F1, unSmoothF1, uv);
                col2 = useBlendingColors(targ_img, col2, F2, unSmoothF2, uv);
            break;
            case DrawMode::COLOR_CLOSEST:
                col1 = useClosestColors(in_img,   F1, uv);
                col2 = useClosestColors(targ_img, F2, uv);
            break;
            default:
            break;
        }
    }

    float fsum = std::fabs(f1) + std::fabs(f2);
    float w1   = std::fabs(f2) / fsum;
    float w2   = std::fabs(f1) / fsum;

    objCol.r = col1.r * w1 + col2.r * w2;
    objCol.g = col1.g * w1 + col2.g * w2;
    objCol.b = col1.b * w1 + col2.b * w2;
    objCol.a = 1.0f;

    return objCol;
}

glm::vec4 STTImethod::useBlendingColors(const cv::Mat &img, glm::vec4 colP, std::vector<float> &fun, float funVal, glm::vec2 uv)
{
    if(funVal < 0.0f)
    {
          float u_x = uv.x, u_y = uv.y;
          int res_x = resX, res_y = resY;
          unsigned char *buff_im = img.data;
          int ch = img.channels();
          size_t step = img.step;
          float *data = fun.data();
          int i, j;
          float R = 0.0f, G = 0.0f, B = 0.0f;
          float total_w = 0.0f;

#ifdef USE_OPENMP
#pragma omp parallel shared(R, G, B, total_w, data, buff_im) private(i, j)
     {
#pragma omp for simd reduction(+:total_w, R, G, B) schedule(static, 5)
#endif
          for(i = 0; i < res_y; ++i)
              for(j = 0; j < res_x; ++j)
              {
                  if(data[j+i*res_x] >= 0.0f)
                  {
                      glm::vec4 col;
                      col.b = buff_im[j*ch+  i*step];
                      col.g = buff_im[j*ch+1+i*step];
                      col.r = buff_im[j*ch+2+i*step];
                      col.a = 1.0f;

                      col = convertFromRGB(col);

                      float x = j / static_cast<float>(res_x);
                      float y = i / static_cast<float>(res_y);
                      float dist = (u_x - x) * (u_x - x) + (u_y - y) * (u_y - y);
                      float distSq = dist*dist;

                      B += col.b / distSq;
                      G += col.g / distSq;
                      R += col.r / distSq;
                      total_w += 1.0f / distSq;
                  }
              }
 #ifdef USE_OPENMP
     }
 #endif
         return glm::vec4(R/(total_w*255.0f), G/(total_w*255.0f), B/(total_w*255.0f), 1.0f);
    }
    return colP;
}

glm::vec4 STTImethod::useClosestColors(const cv::Mat &img, std::vector<float> &fun, glm::vec2 uv)
{
    float u_x = uv.x, u_y = uv.y;
    int res_x = resX, res_y = resY;
    float *data = fun.data();
    float smallest_dist_squared1 = 10000.0f;
    float cl_uv_x, cl_uv_y;
    int i, j;

#ifdef USE_OPENMP
#pragma omp parallel shared(data, cl_uv_x, cl_uv_y) private(i, j)
    {
#pragma omp for simd schedule(static, 5)
#endif
        for(i = 0; i < res_y; ++i)
        {
            for(j = 0; j < res_x; ++j)
            {
               float x = j/static_cast<float>(res_x);
               float y = i/static_cast<float>(res_y);
               float dist_squared = (u_x - x)*(u_x - x) + (u_y - y)*(u_y - y);

               if(data[j+i*res_x] >= 0.0f && dist_squared < smallest_dist_squared1)
               {
                   smallest_dist_squared1 = dist_squared;
                   cl_uv_x = x;
                   cl_uv_y = y;
               }
            }
        }
#ifdef USE_OPENMP
    }
#endif

    unsigned char *buf = img.data;
    glm::vec2 closest_uv(cl_uv_x, cl_uv_y);
    return getPixelColor(buf, img.step, img.channels(), closest_uv);
}

glm::vec4 STTImethod::getPixelColor(const unsigned char *pixBuf, const size_t step, const int numChannels, glm::vec2 uv)
{
    uint r = 0, g = 0, b = 0, a = 0;
    size_t x = static_cast<size_t>(uv.x * resX) * static_cast<size_t>(numChannels);
    size_t y = static_cast<size_t>(uv.y * resY);

    b = pixBuf[x+  y*step];
    g = pixBuf[x+1+y*step];
    r = pixBuf[x+2+y*step];
    a = pixBuf[x+3+y*step];

    glm::vec4 col(r/255.0f, g/255.0f, b/255.0f, a/255.0f);

    return convertFromRGB(col);
}

glm::vec4 STTImethod::convertToRGB(glm::vec4 col)
{
    switch (curColMode)
    {
        case ColorMode::RGB:
            break;
        case ColorMode::HSV:
            col = convertHSVtoRGB(col);
        break;
        case ColorMode::CIELab:
            col = convertCIELabtoRGB(col);
        break;
        default:
        break;
    }
    return col;
}

glm::vec4 STTImethod::convertFromRGB(glm::vec4 col)
{
    switch (curColMode)
    {
        case ColorMode::RGB:
        break;
        case ColorMode::HSV:
            col = convertRGBtoHSV(col);
        break;
        case ColorMode::CIELab:
            col = convertRGBtoCIELab(col);
        break;
        default:
        break;
    }
    return col;
}

glm::vec4 STTImethod::convertRGBtoHSV(glm::vec4 col)
{
    float min, max, delta;
    float h = 0.0f, s = 0.0f, v = 0.0f;

    min = min3(col.r, col.g, col.b);
    max = max3(col.r, col.g, col.b);

    delta = max - min;
    v = max;

    if(max != 0.0f)
        s = delta / max;
    else
    {
        h = 0.0f;
        s = 0.0f;
        v = 0.0f;
    }

    if(col.r == max && col.r != 0.0f)
        h = (col.g - col.b) / delta;        // between yellow & magenta
    else if(col.g == max && col.g != 0.0f)
        h = 2.0f + (col.b - col.r) / delta; // between cyan & yellow
    else if(col.b == max && col.b != 0.0f)
        h = 4.0f + (col.r - col.g) / delta; // between magenta & cyan

    h *= 60.0f;    // degrees
    if(h < 0.0f)
        h += 360.0f;

    if(h > 360.0f)
        h -= 360.0f;

    h /= 360.0f;

    col.r = h;
    col.g = s;
    col.b = v;
    return col;
}

glm::vec4 STTImethod::convertHSVtoRGB(glm::vec4 col)
{
    int i;
    float f, p, q, t;
    float r1, g1, b1;

    if(col.g == 0.0f)
    {
        r1 = g1 = b1 = col.b;
        col.r = col.g = col.b = r1;
    }
    else
    {
        col.r *= 360.0f;
        col.r /= 60.0f;  // sector 0 to 5

        i = static_cast<int>(std::floor(col.r));
        f = col.r - i;              // factorial part of h
        p = col.b * (1.0f - col.b);
        q = col.b * (1.0f - col.b * f);
        t = col.b * (1.0f - col.b * (1.0f - f));

        switch(i)
        {
            case 0:
                r1 = col.b;
                g1 = t;
                b1 = p;
            break;
            case 1:
                r1 = q;
                g1 = col.b;
                b1 = p;
            break;
            case 2:
                r1 = p;
                g1 = col.b;
                b1 = t;
            break;
            case 3:
                r1 = p;
                g1 = q;
                b1 = col.b;
            break;
            case 4:
                r1 = t;
                g1 = p;
                b1 = col.b;
            break;
            default:
                r1 = col.b;
                g1 = p;
                b1 = q;
            break;
      }
      col.r = r1;
      col.g = g1;
      col.b = b1;
    }
    return col;
}

glm::vec4 STTImethod::convertRGBtoCIELab(glm::vec4 col)
{
    //1 step is to convert RGB to XYZ color space
    float r1, g1, b1;
    if(col.r > 0.04045f)
        r1 = std::pow(((col.r+0.055f) / 1.055f), 2.4f);
    else
        r1 = col.r / 12.92f;

    if(col.g > 0.04045f)
        g1 = std::pow(((col.g+0.055f) / 1.055f), 2.4f);
    else
        g1 = col.g / 12.92f;

    if(col.b > 0.04045f)
        b1 = std::pow(((col.b+0.055f) / 1.055f), 2.4f);
    else
        b1 = col.b / 12.92f;

    r1 = r1 * 100.0f;
    g1 = g1 * 100.0f;
    b1 = b1 * 100.0f;

    float X = r1 * 0.4124f + g1 * 0.3576f + b1 * 0.1805f;
    float Y = r1 * 0.2126f + g1 * 0.7152f + b1 * 0.0722f;
    float Z = r1 * 0.0193f + g1 * 0.1192f + b1 * 0.9505f;

    //2 step is to convert XYZ to CIELab color space
    //CIELab is dependong on standart observer parameters for the standart type of light (in our case D65)
    //chosen observer: 2 degree, Illuminant = D65;
    float var_X = X / 95.047f;
    float var_Y = Y / 100.0f;
    float var_Z = Z / 108.883f;

    if (var_X > 0.008856f)
        var_X = std::pow(var_X, 1.0f/3.0f);
    else
        var_X = (7.787f * var_X) + (16.0f / 116.0f);

    if (var_Y > 0.008856f)
        var_Y = std::pow(var_Y, 1.0f/3.0f);
    else
        var_Y = (7.787f * var_Y) + (16.0f / 116.0f);

    if (var_Z > 0.008856f)
        var_Z = std::pow(var_Z, 1.0f/3.0f);
    else
        var_Z = (7.787f * var_Z) + (16.0f / 116.0f);

    col.r = 116.0f * var_Y - 16.0f;
    col.g = 500.0f * (var_X - var_Y);
    col.b = 200.0f * (var_Y - var_Z);
    return col;
}

glm::vec4 STTImethod::convertCIELabtoRGB(glm::vec4 col)
{
    // 1 step: converting CIELab to XYZ color space
    float var_Y = (col.r + 16.0f) / 116.0f;
    float var_X = col.g / 500.0f + var_Y;
    float var_Z = var_Y - col.b / 200.0f;

    if(pow(var_Y, 3.0f) > 0.008856f)
        var_Y = std::pow(var_Y, 3.0f);
    else
        var_Y = (var_Y- 16.0f/116.0f) / 7.787f;

    if(pow(var_X, 3.0f) > 0.008856f)
        var_X = std::pow(var_X, 3.0f);
    else
        var_X = (var_X- 16.0f/116.0f) / 7.787f;

    if(pow(var_Z, 3.0f) > 0.008856f)
        var_Z = std::pow(var_Z, 3.0f);
    else
        var_Z = (var_Z- 16.0f/116.0f) / 7.787f;

    //standart observer: 2 degree + illuminant = D65
    float X = var_X * 95.047f;
    float Y = var_Y * 100.0f;
    float Z = var_Z * 108.883f;

    // 2 step: converting XYZ to RGB color space
    X = X / 100.0f;
    Y = Y / 100.0f;
    Z = Z / 100.0f;

    float var_R = X * 3.2406f    + Y * (-1.5372f) + Z * (-0.4986f);
    float var_G = X * (-0.9686f) + Y * 1.8758f    + Z * 0.0415f;
    float var_B = X * 0.0557f    + Y * (-0.2040f) + Z * 1.0570f;

    if(var_R > 0.0031308f)
        var_R = 1.055f * std::pow(var_R, 1.0f / 2.4f) - 0.055f;
    else
        var_R = 12.92f * var_R;

    if(var_G > 0.0031308f)
        var_G = 1.055f * std::pow(var_G, 1.0f / 2.4f) - 0.055f;
    else
        var_G = 12.92f * var_G;

    if(var_B > 0.0031308f)
        var_B = 1.055f * std::pow(var_B, 1.0f / 2.4f) - 0.055f;
    else
        var_B = 12.92f * var_B;

    col.r = var_R;
    col.g = var_G;
    col.b = var_B;
    return col;
}


glm::vec4 STTImethod::clampColor(glm::vec4 col)
{
    if(col.r <= 0.0f) col.r = 0.0f;
    if(col.g <= 0.0f) col.g = 0.0f;
    if(col.b <= 0.0f) col.b = 0.0f;

    if(col.r >= 1.0f) col.r = 1.0f;
    if(col.g >= 1.0f) col.g = 1.0f;
    if(col.b >= 1.0f) col.b = 1.0f;
    return col;
}

} //namespace hstb
