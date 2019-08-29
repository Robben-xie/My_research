#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H
#include <vector>
#include <list>
#include <opencv/cv.h>

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;//关键点
    cv::Point2i UL, UR, BL, BR; // 二维坐标(坐标值为整型)
    std::list<ExtractorNode>::iterator lit;// 检测角点列表
    bool bNoMore;//标志
};

class ORBextractor
{
public:

    enum {HARRIS_SCORE=0, FAST_SCORE=1 };
    // ORB特征点检测，个数，尺度因子，尺度金字塔中的级别数，快速阈值，最小阈值
    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // 使用八叉树将ORB分散在图像上。
    // Mask is ignored in the current implementation.
    //重载（)运算符，作为提取器的对外接口
    void operator()( cv::InputArray image,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    //为了解决一些频繁调用的小函数大量消耗栈空间（栈内存）的问题，特别的引入了inline修饰符
    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;//倒数
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;//方差
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }
    //图像金字塔，存放各层(采样)图片
    std::vector<cv::Mat> mvImagePyramid;

protected:
    //计算金字塔
    void ComputePyramid(cv::Mat image);
    //计算关键点，并用八叉树的形式存储
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    //八叉树分配
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;//金字塔每层的特征点数

    std::vector<int> umax;//pitch圆的最大坐标

    std::vector<float> mvScaleFactor;//每层相对于原始图像的缩放尺度
    std::vector<float> mvInvScaleFactor;//上面的倒数
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

#endif // ORBEXTRACTOR_H
