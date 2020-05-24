#include "include/distanceTransform2D.h"
#include "include/interpolation2D.h"
#include "include/timer.hpp"

#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

namespace hstb {

SEDTtransform::SEDTtransform(int res_x, int res_y): resX(res_x),
                                                    resY(res_y),
                                                    INF(calculateINF()),
                                                    INTERIOR(INF, INF),
                                                    EXTERIOR(0.0f, 0.0f),
                                                    grid1(resX*resY, INTERIOR),
                                                    grid2(resX*resY, INTERIOR),
                                                    SDT(resX*resY, 0.0f)
{ }

void SEDTtransform::computeDistanceTransform(const cv::Mat &img, bool smooth)
{
    prof::timer timer0;
    timer0.Start();

    initGrids(img);
    calculateDistances(grid1);
    calculateDistances(grid2);
    mergeGrids();
    if(smooth)
    {
        inter2D inter;
        SDT = inter.geoToolsResample2D(&SDT, InterType::GEO_BICUBIC, glm::ivec2(resX, resY),
                                             glm::ivec2(resX, resY), glm::vec2(1, 1), true);
    }

    timer0.End("\nSDT timing: ");
}

void SEDTtransform::initGrids(const cv::Mat &img)
{
    cv::Mat binImg = binariseImage(img);
    unsigned char *img_buff = binImg.data;

    for (int y = 0; y < resY; y++)
       for (int x = 0; x < resX; x++)
       {
          if(img_buff[x+y*resX] > 0)
          {
             assignGridElement(grid1, INTERIOR, x, y);
             assignGridElement(grid2, EXTERIOR, x, y);
          }
          else
          {
             assignGridElement(grid1, EXTERIOR, x, y);
             assignGridElement(grid2, INTERIOR, x, y);
          }
       }
}

void SEDTtransform::compareGridPoints(std::vector<glm::vec2> &grid, glm::vec2 &point,
                                              int offsetX, int offsetY, int x, int y)
{
    if(x + offsetX >= 0 && x + offsetX < resX &&
       y + offsetY >= 0 && y + offsetY < resY)
    {
        glm::vec2 other_point = getGridElement(grid, x+offsetX, y+offsetY);
        other_point.x += offsetX;
        other_point.y += offsetY;

        float dist1 = euclideanDistSq(other_point);
        float dist2 = euclideanDistSq(point);
        if(dist1 < dist2)
            point = other_point;
    }
}

void SEDTtransform::calculateDistances(std::vector<glm::vec2> &grid)
{
    // Pass 0: firstly filtering grid in forward manner on both x & y, then on x in reverse manner
    for (int y = 0; y < resY; y++)
    {
        for (int x = 0; x < resX; x++)
        {
            glm::vec2 point = getGridElement(grid, x, y);
            compareGridPoints(grid, point, -1,  0, x, y);
            compareGridPoints(grid, point,  0, -1, x, y);
            compareGridPoints(grid, point, -1, -1, x, y);
            compareGridPoints(grid, point,  1, -1, x, y);
            assignGridElement(grid, point, x, y);
        }

        for (int x = resX - 1; x >= 0; x--)
        {
            glm::vec2 point = getGridElement(grid, x, y);
            compareGridPoints(grid, point, 1, 0, x, y);
            assignGridElement(grid, point, x, y);
        }
    }

    // Pass 1: firstly filtering grid in reverse manner on both x & y, then on x in forward manner
    for (int y = resY-1; y >= 0; y--)
    {
       for (int x = resX-1; x >= 0; x--)
       {
            glm::vec2 point = getGridElement(grid, x, y);
            compareGridPoints(grid, point,  1,  0, x, y);
            compareGridPoints(grid, point,  0,  1, x, y);
            compareGridPoints(grid, point, -1,  1, x, y);
            compareGridPoints(grid, point,  1,  1, x, y);
            assignGridElement(grid, point, x, y);
       }

       for (int x = 0; x < resX; x++)
       {
           glm::vec2 point = getGridElement(grid, x, y);
           compareGridPoints(grid, point, -1, 0, x, y);
           assignGridElement(grid, point, x, y);
       }
    }
}

void SEDTtransform::mergeGrids()
{
    SDT.clear(); SDT.resize(resX*resY);

    for(int y = 0; y < resY; y++)
        for(int x = 0; x < resX; x++)
        {
            float dist1 = std::sqrt(euclideanDistSq(getGridElement(grid1, x, y)));
            float dist2 = std::sqrt(euclideanDistSq(getGridElement(grid2, x, y)));
            SDT[x + y * resX] = (dist1 - dist2) * 10.0f / INF;
        }
}

cv::Mat SEDTtransform::binariseImage(const cv::Mat &img)
{
    cv::Mat srcGray, binImg;
    cv::cvtColor(img, srcGray, cv::COLOR_BGR2GRAY, 1);
    cv::threshold(srcGray, binImg, 0, 255, cv::THRESH_BINARY);
    return binImg;
}

} //namespace hstb
