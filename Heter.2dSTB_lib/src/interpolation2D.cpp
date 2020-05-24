#include "include/interpolation2D.h"

#include "Mathematics/IntpAkimaUniform2.h"
#include "Mathematics/IntpBicubic2.h"
#include "Mathematics/IntpBilinear2.h"
#include "Mathematics/IntpBSplineUniform.h"
#include "Mathematics/IntpThinPlateSpline2.h"

#include <iostream>

namespace hstb {

std::vector<float> inter2D::geoToolsResample2D(std::vector<float> *field, InterType type,
                                               glm::ivec2 inRes, glm::ivec2 outRes,
                                               glm::vec2 step, bool precise)
{
    glm::vec2 stP(0, 0);
    if(type == InterType::GEO_AKIMA)
    {
        std::vector<float> result;
        gte::IntpAkimaUniform2<float> akimaSp(inRes.x, inRes.y, stP.x, step.x,
                                              stP.y, step.y, &field->at(0));
        for(int y = stP.y; y < outRes.y; y++)
            for(int x = stP.x; x < outRes.x; x++)
                result.push_back(akimaSp(x, y));
        return result;
    }
    else if(type == InterType::GEO_BICUBIC)
    {
        std::vector<float> result;
        gte::IntpBicubic2<float> bicubicSp(inRes.x, inRes.y, stP.x, step.x,
                                           stP.y, step.y, &field->at(0), precise);
        for(int y = stP.y; y < outRes.y; y++)
            for(int x = stP.x; x < outRes.x; x++)
                result.push_back(bicubicSp(x, y));
        return result;
    }
    else if(type == InterType::GEO_THINPLATE)
    {
        std::vector<float> result, vecX, vecY;
        for(int i = 0; i < inRes.x; i++)
        {
            vecX.push_back(i);
            vecY.push_back(i);
        }

        gte::IntpThinPlateSpline2<float> thinPlateSp(field->size(), &vecX[0], &vecY[0], &field->at(0), 0.0, false);

        for(int y = stP.y; y < outRes.y; y++)
            for(int x = stP.x; x < outRes.x; x++)
                result.push_back(thinPlateSp(x, y));
        return result;
    }
    else if(type == InterType::GEO_BILINEAR)
    {
        std::vector<float> result;
        gte::IntpBilinear2<float> bilinearSp(inRes.x, inRes.y, stP.x, step.x,
                                             stP.y, step.y, &field->at(0));
        for(int y = stP.y; y < outRes.y; y++)
            for(int x = stP.x; x < outRes.x; x++)
                result.push_back(bilinearSp(x, y));
        return result;
    }
    else
    {
        std::wcerr << "Warning <GeoToolsResample2D(), InterType>: Unknown option!" << std::endl;
        std::wcerr << "Returing input field!" << std::endl;
        return *field;
    }
}

}//namespace frep2D

