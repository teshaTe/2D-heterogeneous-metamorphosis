#ifndef H_DISTANCE_TRANSFORM_CLASS
#define H_DISTANCE_TRANSFORM_CLASS

#include <opencv4/opencv2/opencv.hpp>
#include <glm/glm.hpp>

#include <vector>

namespace metamorphosis {

struct Point
{
  Point(float dx, float dy);
  float x, y;
  float square_distance();
};

class distanceTransform
{
public:
    distanceTransform( cv::Mat *inputImg, cv::Mat *targetImg, int b_sh );
    distanceTransform( cv::Mat *img, int b_sh );
    ~distanceTransform() {}

    void smoothDistanceTransformField(std::vector<float> *inField , std::vector<float> *outField);

    inline std::vector<float> getSDDT1() { return SDDT1; }
    inline std::vector<float> getSDDT2() { return SDDT2; }

    void inverseMappingResult(float *res, std::vector<float> SDF, float u_shifted, float v_shifted, cv::Mat affine_m);

private:
    inline float getGridElement (float *grid, int x, int y, int width) { return grid[x+y*width]; }
    inline float extrapolateVals(float val1, float val2, int n1, int n2, int cur_n) { return val1 + (val2 - val1) * (cur_n - n1)/(n2 - n1); }

private:
    void generateDistanceTransform( std::vector<float> &field, cv::Mat *img );
    void initGrids(cv::Mat *img );
    void compareGridPoints( std::vector<Point> &grid, Point &point, int offsetX, int offsetY, int x, int y );
    void calculateDistances(std::vector<Point> &grid );
    void mergeGrids( std::vector<float> &grid );

    cv::Mat binariseImage( cv::Mat *src );

    inline int clipWithBounds( int n, int n_min, int n_max ) { return n > n_max ? n_max :( n < n_min ? n_min : n ); }
    inline int calculateINF() { return std::max( resX, resY ) + 1; }
    inline Point getGridElement( std::vector<Point> &grid, int posX, int posY ) { return grid[posX+posY*resX]; }
    inline void assignGridElement( std::vector<Point> &grid, Point point, int posX, int posY ) { grid[posX+posY*resX] = point; }

private:
    cv::Mat inv_affine_m;
    int bShift, resX, resY;

    const int INF;
    const Point INSIDE;
    const Point OUTSIDE;

    std::vector<Point> grid1;
    std::vector<Point> grid2;

    std::vector<float> SDDT1, SDDT2;
};

} // namespace metamorphosis
#endif // H_DISTANCE_TRANSFORM_CLASS
