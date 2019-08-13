#include "include/colourInterpolation.h"

namespace metamorphosis {

colourInter::colourInter(cv::Mat initIm, cv::Mat targIm, std::vector<float> f1, std::vector<float> f2,
                         int width, int height, DrawMode dMode, ColorMode cMode, int colSegmNum)
{
    inImg   = initIm;
    targImg = targIm;

    initImBuf = inImg.data;
    targImBuf = targIm.data;
    F1 = f1;
    F2 = f2;

    resX = width;
    resY = height;

    curDrawMode = dMode;
    curColMode  = cMode;

    colSegm = std::make_shared<colourSegm>( resX, resY, colSegmNum );
}

glm::vec4 colourInter::computeSTTI_noCorrespondences(float f1, float f2, float unSmoothF1, float unSmoothF2, glm::vec2 uv)
{
    glm::vec4 objCol( 0.0f, 0.0f, 0.0f, 1.0f );
    glm::vec4 zero( 0.0, 0.0, 0.0, 1.0 );

    glm::vec4 col1 = getPixelColor( initImBuf, inImg.step,   inImg.channels(),   uv );
    glm::vec4 col2 = getPixelColor( targImBuf, targImg.step, targImg.channels(), uv );

    if( col1 == zero || col2 == zero )
    {
        switch ( curDrawMode )
        {
            case DrawMode::WHITE:
              if(col1 == zero) col1 = glm::vec4( 1.0, 1.0, 1.0, 1.0 );
              if(col2 == zero) col2 = glm::vec4( 1.0, 1.0, 1.0, 1.0 );
              break;
            case DrawMode::SIMPLE_BLEND:
              useBlendingColors( &col1, inImg,   &F1, unSmoothF1, uv );
              useBlendingColors( &col2, targImg, &F2, unSmoothF2, uv );
              break;
            case DrawMode::SIMPLE_CLOSEST:
              useClosestColors( &col1, inImg,   &F1, uv );
              useClosestColors( &col2, targImg, &F2, uv );
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

glm::vec4 colourInter::computeSTTI_withCorrespondences(float f1, float f2, float unSmoothF1, float unSmoothF2, glm::vec2 uv)
{

}

void colourInter::constructCorrespondences( cv::Mat img1, cv::Mat img2 )
{
    im1Colors = colSegm.get()->detectDominantColors( img1 );
    classificator1 = colSegm.get()->getPixelClassificator();
    root1 = colSegm.get()->getRootNode();

    im2Colors = colSegm.get()->detectDominantColors( img2 );
    classificator2 = colSegm.get()->getPixelClassificator();
    root2 = colSegm.get()->getRootNode();
}

glm::vec4 colourInter::getPixelColor(unsigned char *pixBuf, size_t step, int numChannels, glm::vec2 uv)
{
    uint r = 0, g = 0, b = 0, a = 0;

    size_t x = static_cast<size_t>(uv.x * resX) * static_cast<size_t>( numChannels );
    size_t y = static_cast<size_t>(uv.y * resY);

    b = pixBuf[x+  y*step];
    g = pixBuf[x+1+y*step];
    r = pixBuf[x+2+y*step];
    a = pixBuf[x+3+y*step];

    glm::vec4 col( static_cast<float>(r/255.0f), static_cast<float>(g/255.0f),
                   static_cast<float>(b/255.0f), static_cast<float>(a/255.0f) );
    convertFromRGB( &col.r, &col.g, &col.b );
    return col;
}

void colourInter::useBlendingColors( glm::vec4 *col, cv::Mat img, std::vector<float> *fun, float funVal, glm::vec2 uv )
{
    if( funVal < 0.0f )
    {
         float u_x = uv.x, u_y = uv.y;
         int res_x = resX, res_y = resY;
         unsigned char *buff_im = img.data;

         float total_w = 0.0f;
         float R = 0.0f, G = 0.0f, B = 0.0f;
         int ch = img.channels();
         size_t step = img.step;
         float *data = fun->data();
         float r, g, b, x, y, dist, distSq;
         int i, j;

#ifdef USE_OPENMP
#pragma omp parallel shared( R, G, B, total_w, data, buff_im, ch, step, res_x, res_y ) private( r, g, b, x, y, i, j, dist, distSq)
    {
#pragma omp for simd reduction( +:total_w, R, G, B) schedule( static, 5 )
#endif
         for( i = 0; i < res_y; ++i)
         {
             for( j = 0; j < res_x; ++j)
             {
                 if( data[j+i*res_x] >= 0.0f )
                 {
                     b = buff_im[j*ch+  i*step] / 255.0f;
                     g = buff_im[j*ch+1+i*step] / 255.0f;
                     r = buff_im[j*ch+2+i*step] / 255.0f;
                     convertFromRGB( &r, &g, &b );

                     x = static_cast<float>(j) / static_cast<float>(res_x);
                     y = static_cast<float>(i) / static_cast<float>(res_y);
                     dist = (u_x - x) * (u_x - x) + (u_y - y) * (u_y - y);
                     distSq = dist*dist;

                     B += b / distSq;
                     G += g / distSq;
                     R += r / distSq;
                     total_w += 1.0f / distSq;
                 }
             }
         }
#ifdef USE_OPENMP
    }
#endif
        *col = glm::vec4( R/total_w, G/total_w, B/total_w, 1.0f );
    }
}

void colourInter::useClosestColors(glm::vec4 *col, cv::Mat img, std::vector<float> *fun, glm::vec2 uv)
{
    float u_x = uv.x, u_y = uv.y;
    int res_x = resX, res_y = resY;

    float *data = fun->data();
    float smallest_dist_squared1 = 10000.0f;
    float cl_uv_x, cl_uv_y, dist_squared, x, y;
    int i, j;

#ifdef USE_OPENMP
#pragma omp parallel shared( data,cl_uv_x,cl_uv_y,res_x,res_y,smallest_dist_squared1,u_x,u_y) private(x,y,i,j,dist_squared)
    {
#pragma omp for simd schedule(static, 5)
#endif
        for( i = 0; i < res_y; ++i)
        {
            for( j = 0; j < res_x; ++j)
            {
               x = static_cast<float>(j)/static_cast<float>(res_x);
               y = static_cast<float>(i)/static_cast<float>(res_y);
               dist_squared = (u_x - x)*(u_x - x) + (u_y - y)*(u_y - y);

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
    *col = getPixelColor( buf, img.step, img.channels(), closest_uv );
}

void colourInter::convertFromRGB(float *r, float *g, float *b)
{
    switch (curColMode)
    {
      case ColorMode::RGB:
        break;
      case ColorMode::HSV:
        convertRGBtoHSV( r, g, b );
        break;
      case ColorMode::CIELab:
        convertRGBtoCIELab( r, g, b );
        break;
      default:
        break;
    }
}

void colourInter::convertRGBtoHSV(float *r, float *g, float *b)
{
    float min, max, delta;
    float h = 0.0f, s = 0.0f, v = 0.0f;

    min = min3( *r, *g, *b );
    max = max3( *r, *g, *b );

    delta = max - min;
    v     = max;

    if( max != 0.0f )
      s = delta / max;
    else
    {
      h = 0.0f;
      s = 0.0f;
      v = 0.0f;
    }

    if( *r == max && *r != 0.0f )
      h = ( *g - *b ) / delta;        // between yellow & magenta

    else if( *g == max && *g != 0.0f )
      h = 2.0f + ( *b - *r ) / delta;    // between cyan & yellow

    else if( *b == max && *b != 0.0f )
      h = 4.0f + ( *r - *g ) / delta;    // between magenta & cyan

    h *= 60.0f;    // degrees
    if( h < 0.0f )
      h += 360.0f;

    if( h > 360.0f )
      h -= 360.0f;

    h /= 360.0f;

    *r = h;
    *g = s;
    *b = v;
}

void colourInter::convertRGBtoCIELab(float *r, float *g, float *b)
{
    //1 step is to convert RGB to XYZ color space
    float r1, g1, b1;
    if(*r > 0.04045f) r1 = std::pow(((*r+0.055f) / 1.055f), 2.4f);
    else                    r1 = *r / 12.92f;

    if(*g > 0.04045f) g1 = std::pow(((*g+0.055f) / 1.055f), 2.4f);
    else                    g1 = *g / 12.92f;

    if(*b > 0.04045f) b1 = std::pow(((*b+0.055f) / 1.055f), 2.4f);
    else                    b1 = *b / 12.92f;

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

    if ( var_X > 0.008856f ) var_X = std::pow(var_X, 1.0f/3.0f );
    else                     var_X = ( 7.787f * var_X ) + ( 16.0f / 116.0f );

    if ( var_Y > 0.008856f ) var_Y = std::pow(var_Y, 1.0f/3.0f );
    else                     var_Y = ( 7.787f * var_Y ) + ( 16.0f / 116.0f );

    if ( var_Z > 0.008856f ) var_Z = std::pow(var_Z, 1.0f/3.0f );
    else                     var_Z = ( 7.787f * var_Z ) + ( 16.0f / 116.0f );

    *r = 116.0f * var_Y - 16.0f;
    *g = 500.0f * (var_X - var_Y);
    *b = 200.0f * (var_Y - var_Z);
}

void colourInter::convertToRGB(float *r, float *g, float *b)
{
    switch ( curColMode )
    {
      case ColorMode::RGB:
        break;
      case ColorMode::HSV:
        convertHSVtoRGB( r, g, b );
        break;
      case ColorMode::CIELab:
        convertCIELabToRGB( r, g, b );
        break;
      default:
        break;
    }
}

void colourInter::convertHSVtoRGB(float *r, float *g, float *b)
{
    int i;
    float f, p, q, t;
    float r1, g1, b1;

    if( *g == 0.0f )
    {
      r1 = g1 = b1 = *b;
      *r = *g = *b = r1;
    }
    else
    {
      *r *= 360.0f;
      *r /= 60.0f;  // sector 0 to 5

      i = static_cast<int>(std::floor( *r ));
      f = *r - i;              // factorial part of h
      p = *b * ( 1.0f - *b );
      q = *b * ( 1.0f - *b * f );
      t = *b * ( 1.0f - *b * ( 1.0f - f ) );

      switch( i )
      {
        case 0:
          r1 = *b;
          g1 = t;
          b1 = p;
          break;
        case 1:
          r1 = q;
          g1 = *b;
          b1 = p;
          break;
        case 2:
          r1 = p;
          g1 = *b;
          b1 = t;
          break;
        case 3:
          r1 = p;
          g1 = q;
          b1 = *b;
          break;
        case 4:
          r1 = t;
          g1 = p;
          b1 = *b;
          break;
        default:
          r1 = *b;
          g1 = p;
          b1 = q;
          break;
      }

      *r = r1;
      *g = g1;
      *b = b1;
    }
}

void colourInter::convertCIELabToRGB(float *r, float *g, float *b)
{
    // 1 step: converting CIELab to XYZ color space
    float var_Y = (*r + 16.0f) / 116.0f;
    float var_X = *g / 500.0f + var_Y;
    float var_Z = var_Y - *b / 200.0f;

    if(pow(var_Y, 3.0f) > 0.008856f) var_Y = std::pow(var_Y, 3.0f);
    else                             var_Y = (var_Y- 16.0f/116.0f) / 7.787f;

    if(pow(var_X, 3.0f) > 0.008856f) var_X = std::pow(var_X, 3.0f);
    else                             var_X = (var_X- 16.0f/116.0f) / 7.787f;

    if(pow(var_Z, 3.0f) > 0.008856f) var_Z = std::pow(var_Z, 3.0f);
    else                             var_Z = (var_Z- 16.0f/116.0f) / 7.787f;

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

    if ( var_R > 0.0031308f ) var_R = 1.055f * std::pow( var_R, 1.0f / 2.4f ) - 0.055f;
    else                      var_R = 12.92f * var_R;
    if ( var_G > 0.0031308f ) var_G = 1.055f * std::pow( var_G, 1.0f / 2.4f ) - 0.055f;
    else                      var_G = 12.92f * var_G;
    if ( var_B > 0.0031308f ) var_B = 1.055f * std::pow( var_B, 1.0f / 2.4f ) - 0.055f;
    else                      var_B = 12.92f * var_B;

    *r = var_R;
    *g = var_G;
    *b = var_B;
}

void colourInter::clampColor(glm::vec4 *col)
{
    if(col->r <= 0.0f) col->r = 0.0f;
    if(col->g <= 0.0f) col->g = 0.0f;
    if(col->b <= 0.0f) col->b = 0.0f;

    if(col->r >= 1.0f) col->r = 1.0f;
    if(col->g >= 1.0f) col->g = 1.0f;
    if(col->b >= 1.0f) col->b = 1.0f;
}


} //namespace metamorphosis
