#ifndef H_DISTANCE_TRANSFORM_CLASS
#define H_DISTANCE_TRANSFORM_CLASS

/*This algorithm is based on the following paper:
 * @article{Leymarie1992FastRS,
            title={Fast raster scan distance propagation on the discrete rectangular lattice},
                   author={Frederic Fol Leymarie and Martin D. Levine},
                   journal={CVGIP Image Underst.},
                   year={1992},
                   volume={55},
                   doi = {https://doi.org/10.1016/1049-9660%2892%2990008-Q}
                   pages={84-94}
    }
 */

#include <opencv4/opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include <vector>

namespace hstb {

class SEDTtransform
{
public:
    SEDTtransform(int res_x, int res_y);
    ~SEDTtransform() { }

    void computeDistanceTransform(const cv::Mat &img, bool smooth);

    inline std::vector<float> getSDT()  { return SDT; }
    inline void setNewResolution(int res_x, int res_y) { resX = res_x; resY = res_y; }

private:
    void initGrids(const cv::Mat &img);
    void compareGridPoints(std::vector<glm::vec2> &grid, glm::vec2 &point, int offsetX, int offsetY, int x, int y);
    void calculateDistances(std::vector<glm::vec2> &grid);
    void mergeGrids();

    cv::Mat binariseImage(const cv::Mat &img);

    inline float euclideanDistSq(glm::vec2 p) { return p.x*p.x + p.y*p.y; }
    inline float getGridElement (float *grid, int x, int y, int stride) { return grid[x+y*stride]; }
    inline float extrapolateVals(float val1, float val2, int n1, int n2, int cur_n) { return val1 + (val2 - val1) * (cur_n - n1)/(n2 - n1); }
    inline int clipWithBounds(int n, int n_min, int n_max) { return n > n_max ? n_max :(n < n_min ? n_min : n); }
    inline int calculateINF() { return std::max(resX, resY) + 1; }
    inline glm::vec2 getGridElement(std::vector<glm::vec2> &grid, int posX, int posY) { return grid[posX+posY*resX]; }
    inline void assignGridElement(std::vector<glm::vec2> &grid, glm::vec2 point, int posX, int posY) { grid[posX+posY*resX] = point; }

private:
    int resX, resY;
    int INF;
    glm::vec2 INTERIOR;
    glm::vec2 EXTERIOR;
    cv::Mat inv_affine_m;
    std::vector<glm::vec2> grid1;
    std::vector<glm::vec2> grid2;
    std::vector<float> SDT;
};


} //namespace hstb
#endif //H_DISTANCE_TRANSFORM_CLASS
