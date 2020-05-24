#ifndef H_INTERPOLATION_2D_CLASS
#define H_INTERPOLATION_2D_CLASS

/*
 * This class relies on the Geometric Tools library for handling smoothing operations
 * using 4 interpolation types.
 * https://www.geometrictools.com/
 */

#include <vector>
#include <glm/glm.hpp>

namespace hstb {

enum class InterType
{
    GEO_AKIMA,
    GEO_BICUBIC,
    GEO_BILINEAR,
    GEO_THINPLATE
};

class inter2D
{
public:
    inter2D() {}
    ~inter2D() {}

    std::vector<float> geoToolsResample2D(std::vector<float> *field, InterType type, glm::ivec2 inRes,
                                                glm::ivec2 outRes, glm::vec2 step, bool precise=true);
};

} //namespace hstb
#endif
