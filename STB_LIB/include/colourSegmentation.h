// this code is based on the tutorial http://www.aishack.in/tutorials/dominant-color/
// by Utkarsh Sinha

#ifndef H_CLASS_COLOUR_SEGMENTATION
#define H_CLASS_COLOUR_SEGMENTATION

#include <opencv4/opencv2/opencv.hpp>
#include <queue>
#include <vector>

namespace metamorphosis {

// a single node structure of the tree structure which will be further used for
// segmentation of the image into colour regions
typedef struct colNode
{
    cv::Mat mean;
    cv::Mat covar;
    uchar classID;

    colNode *left;
    colNode *right;
} colNode;

class colourSegm
{
public:
    colourSegm(int im_w, int im_h , int colNumber);
    ~colourSegm();

    std::vector<cv::Vec3b> detectDominantColors( cv::Mat img );
    std::vector<cv::Vec3b> getDominantColors(colNode *root );
    std::vector<colNode *> getTreeLeaves(colNode *root );

    cv::Mat getQuantizedImg( colNode *root );
    cv::Mat getSegmentedImage();
    cv::Mat getDominantPalette( std::vector<cv::Vec3b> colors );

    inline colNode* getRootNode() { return root; }
    inline cv::Mat  getPixelClassificator() { return pixClasses; }

private:
    colNode *getMaxEigenValNode( colNode *node );
    void     getPixClassMeanCov( cv::Mat img, colNode *node );
    void     definePartitionClass(cv::Mat img, uchar nextID, colNode *node );

    uchar getNextClassID(colNode *root );

private:
    const int width, height;
    int maxColCount, colNum;
    cv::Mat pixClasses;
    colNode *root;

};

} //namespace metamorphosis

#endif //H_CLASS_COLOUR_SEGMENTATION
