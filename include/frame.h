#ifndef FRAME_H
#define FRAME_H

#include <vector>

#include <./BowVector.h>
#include <./FeatureVector.h>

#include "mappoint.h"
#include "orbvocabulary.h"
#include "keyframe.h"
#include "orbextractor.h"

#include <opencv2/opencv.hpp>

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef);


    // Extract ORB on the image. 0 for left image and 1 for right image.
    // 提取的关键点存放在mvKeys和mDescriptors中
    // ORB是直接调orbExtractor提取的
    void ExtractORB(const cv::Mat &im);

    // Compute Bag of Words representation.
    // 存放在mBowVec中
    void ComputeBoW();

    // Set the camera pose.
    // 用Tcw更新mTcw
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter()
    {
        return mOw.clone();
    }

    // Returns inverse of rotation旋转矩阵的逆
    inline cv::Mat GetRotationInverse()
    {
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    // 判断路标点是否在视野中
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft;

    // Frame timestamp.时间㭫
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Number of KeyPoints.
    int N; //< KeyPoints数量

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // mvKeys:原始左图像提取出的特征点（未校正）
    // mvKeysUn:校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    // 左目摄像头和右目摄像头特征点对应的描述子
    cv::Mat mDescriptors;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // 每个特征点对应的MapPoint
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    // 观测不到Map中的3D点
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子，降低匹配复杂度
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵Pw-Pc

    // Current and Next Frame id.
    static long unsigned int nNextId; ///< Next Frame id.下一帧的序号
    long unsigned int mnId; ///< Current Frame id.当前帧的序号

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;//指针，指向参考关键帧

    // Scale pyramid info.
    int mnScaleLevels;//图像金字塔的层数
    float mfScaleFactor;//图像金字塔的尺度因子
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    // 用于确定画格子时的边界
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &image);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw; ///< Rotation from world to camera
    cv::Mat mtcw; ///< Translation from world to camera
    cv::Mat mRwc; ///< Rotation from camera to world
    cv::Mat mOw;  ///< mtwc,Translation from camera to world Pc-Pw
};
#endif // FRAME_H

