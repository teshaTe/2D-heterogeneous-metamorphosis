#include "include/colourSegmentation.h"
#include <iostream>
#include <random>

namespace metamorphosis {

colourSegm::colourSegm(int im_w, int im_h, int colNumber): width(im_w), height(im_h)
{
    //defining the data structure for storing the tree structure

    root = new colNode();
    root->classID = 1;             //index for the initial node during classification of the pixels
    root->left  = nullptr;         // no children specified
    root->right = nullptr;         // no children specified
    pixClasses = cv::Mat( im_h, im_w, CV_8UC1, cv::Scalar(1) );

    colNum = colNumber;
    maxColCount = colNumber*2;
}

colourSegm::~colourSegm(){}

std::vector<cv::Vec3b> colourSegm::detectDominantColors( cv::Mat img )
{
    colNode *next = root;
    getPixClassMeanCov( img, root ); // obtain mean and covariance statistics for the image pixels

    for (int i = 0; i < colNum - 1; i++)
    {
        next = getMaxEigenValNode( root );                         // looking for the node with the biggest eigenvalue
        definePartitionClass( img, getNextClassID( root ), next ); // updating the pixClasses matrix
        getPixClassMeanCov( img, next->left );                     // calculating statistics for the new left node
        getPixClassMeanCov( img, next->right );                    // caclualting statistics for the new right node
    }

    std::vector<cv::Vec3b> colors = getDominantColors( root );
    return colors;
}

void colourSegm::definePartitionClass(cv::Mat img, uchar nextID, colNode *node)
{
    const int classID       = node->classID;
    const uchar newID_left  = nextID;
    const uchar newID_right = nextID + 1;

    cv::Mat mean  = node->mean;
    cv::Mat covar = node->covar;

    cv::Mat eigenValues, eigenVectors;
    cv::eigen( covar, eigenValues, eigenVectors );

    cv::Mat eig = eigenVectors.row(0);
    cv::Mat comparisonVal = eig * mean; // this is kind of threshold

    node->left  = new colNode();
    node->right = new colNode();
    node->left->classID  = newID_left;
    node->right->classID = newID_right;

    unsigned char *imBuff = img.data;
    unsigned char *classBuff = pixClasses.data;

    for ( int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if( classBuff[x*pixClasses.channels()+y*pixClasses.step] != classID )
                continue;

            cv::Mat scaled = cv::Mat( 3, 1, CV_64FC1, cv::Scalar(0) );
            scaled.at<double>(0) = imBuff[x*img.channels()+y*img.step]   / 255.0f;
            scaled.at<double>(1) = imBuff[x*img.channels()+1+y*img.step] / 255.0f;
            scaled.at<double>(2) = imBuff[x*img.channels()+2+y*img.step] / 255.0f;

            cv::Mat curVal = eig * scaled;
            if( curVal.at<double>(0, 0) <= comparisonVal.at<double>(0, 0))
            {
                classBuff[x*pixClasses.channels()+y*pixClasses.step] = newID_left;
            }
            else {
                classBuff[x*pixClasses.channels()+y*pixClasses.step] = newID_right;
            }
        }
    }
}

void colourSegm::getPixClassMeanCov(cv::Mat img, colNode *node)
{
    const uchar classID = node->classID;

    //creating matrices for storing mean and covarinace data
    cv::Mat mean  = cv::Mat( 3, 1, CV_64FC1, cv::Scalar(0) );
    cv::Mat covar = cv::Mat( 3, 3, CV_64FC1, cv::Scalar(0) );

    double pixCount = 0.0;
    unsigned char *imBuff = img.data;
    unsigned char *classBuff = pixClasses.data;

    for ( int y = 0; y < height; y++ )
    {
        for (int x = 0; x < width; x++)
        {
            if( classBuff[x*pixClasses.channels()+y*pixClasses.step] != classID )
                continue;

            cv::Mat scaled  = cv::Mat( 3, 1, CV_64FC1, cv::Scalar(0) );

            scaled.at<double>(0) = imBuff[x*img.channels()  +y*img.step] / 255.0f;
            scaled.at<double>(1) = imBuff[x*img.channels()+1+y*img.step] / 255.0f;
            scaled.at<double>(2) = imBuff[x*img.channels()+2+y*img.step] / 255.0f;

            mean  += scaled;
            covar  = covar + ( scaled * scaled.t() );
            pixCount++;
        }
    }

    covar = covar - ( mean * mean.t() ) / pixCount;
    mean  = mean / pixCount;

    node->mean  = mean.clone();
    node->covar = covar.clone();
}

std::vector<cv::Vec3b> colourSegm::getDominantColors( colNode *root)
{
    std::vector<colNode*> leaves = getTreeLeaves( root );
    std::vector<cv::Vec3b> ret;

    for (size_t i = 0; i < leaves.size(); i++)
    {
        cv::Mat mean = leaves[i]->mean;
        ret.push_back( cv::Vec3b( mean.at<double>(0) * 255.0f,
                                  mean.at<double>(1) * 255.0f,
                                  mean.at<double>(2) * 255.0f));
    }

    return ret;
}

colNode *colourSegm::getMaxEigenValNode(colNode *node)
{
    double maxEigen = -1.0;

    cv::Mat eigenValues, eigenVectors;
    std::queue<colNode*> queue;

    queue.push( node );
    colNode *ret = node;

    //node without children has the maximum eigenvalue
    if( !node->left && !node->right )
        return node;

    //breadth-first traversal of the constructed tree
    // visiting the elemnts level-by-level, the node is always split into two subnodes
    while( queue.size() > 0 )
    {
        colNode  *qNode = queue.front();
        queue.pop();

        if( qNode->left && qNode->right )
        {
            queue.push( qNode->left );
            queue.push( qNode->right );
            continue;
        }

        // if the node is a leaf node, we calculate the eigenvalues of the covariance matrix;
        // cv::eigen() returns eigen values in a descending order
        cv::eigen( qNode->covar, eigenValues, eigenVectors );
        double val = eigenValues.at<double>(0);
        if( val > maxEigen )
        {
            maxEigen = val;
            ret = qNode;
        }
    }

    return ret;
}

uchar colourSegm::getNextClassID(colNode *root)
{
    int maxID = 0;
    std::queue<colNode*> queue;

    queue.push(root);

    while (queue.size() > 0)
    {
        colNode *current = queue.front();
        queue.pop();

        if( current->classID > maxID )
            maxID = current->classID;
        if( current->left != nullptr )
            queue.push( current->left );
        if( current->right != nullptr )
            queue.push( current->right );
    }
    return maxID + 1;
}

std::vector<colNode*> colourSegm::getTreeLeaves(colNode *root)
{
    std::vector<colNode*> ret;
    std::queue<colNode*> queue;

    queue.push( root );

    while( queue.size() > 0 )
    {
        colNode *current = queue.front();
        queue.pop();

        if( current->left && current->right )
        {
            queue.push(current->left);
            queue.push(current->right);
            continue;
        }

        ret.push_back( current );
    }
    return ret;
}

cv::Mat colourSegm::getQuantizedImg(colNode *root)
{
    std::vector<colNode*> leaves = getTreeLeaves(root);
    cv::Mat ret( height, width, CV_8UC3, cv::Scalar(0) );

    unsigned char *retBuff   = ret.data;
    unsigned char *classBuff = pixClasses.data;

    for( int y = 0; y < height; y++ )
    {
        for ( int x = 0; x < width; x++ )
        {
            for (size_t i = 0; i < leaves.size(); i++)
            {
                if( leaves[i]->classID == classBuff[x*pixClasses.channels()+y*pixClasses.step] )
                {
                    retBuff[x*ret.channels()+y*ret.step]   = leaves[i]->mean.at<double>(0) * 255;
                    retBuff[x*ret.channels()+1+y*ret.step] = leaves[i]->mean.at<double>(1) * 255;
                    retBuff[x*ret.channels()+2+y*ret.step] = leaves[i]->mean.at<double>(2) * 255;
                }
            }
        }
    }
    return ret;
}

cv::Mat colourSegm::getSegmentedImage()
{
    cv::Vec3b *palette = new cv::Vec3b[ maxColCount ];

    std::default_random_engine generator;
    std::uniform_int_distribution<int>distribution(0, 255);

    for (size_t i = 0; i < maxColCount; i++)
    {
        int r = distribution(generator);
        int g = distribution(generator);
        int b = distribution(generator);

        palette[i] = cv::Vec3b( r, g, b );
    }

    cv::Mat ret = cv::Mat( height, width, CV_8UC3 );
    unsigned char *colBuff      = ret.data;
    unsigned char *pixClassBuff = pixClasses.data;

    for (int y = 0; y < height; y++)
    {
        for ( int x = 0; x < width; x++)
        {
            int color = pixClassBuff[x*pixClasses.channels()+y*pixClasses.step];
            if( color >= maxColCount )
            {
                std::cerr << "the amount of predefined colors should be increased! " << std::endl;
                exit(-1);
            }
            colBuff[x*ret.channels()+y*ret.step]     = palette[color][0];
            colBuff[x*ret.channels()+ 1+ y*ret.step] = palette[color][1];
            colBuff[x*ret.channels()+ 2+ y*ret.step] = palette[color][2];
        }
    }

    return ret;
}

cv::Mat colourSegm::getDominantPalette(std::vector<cv::Vec3b> colors)
{
    const int tileSize = 64;
    cv::Mat ret = cv::Mat( tileSize, tileSize * colors.size(), CV_8UC3, cv::Scalar(0));

    for ( int i = 0; i < colors.size(); i++)
    {
        cv::Rect rect( i*tileSize, 0, tileSize, tileSize );
        cv::rectangle( ret, rect, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]), cv::FILLED );
    }

    return ret;
}

} // namespace metamorphosis
