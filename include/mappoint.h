#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "keyframe.h"
#include "frame.h"
#include "map.h"

#include <opencv2/core/core.hpp>
#include<mutex>

class KeyFrame;
class Map;
class Frame;

/**
 * @brief MapPoint是一个地图点
 */
class MapPoint
{
public:
    // 两种构造函数，用于初始化字段
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    // 设置地图点的世界坐标
    void SetWorldPos(const cv::Mat &Pos);
    // 获得地图点的世界坐标
    cv::Mat GetWorldPos();

    /**
     * @brief GetNormal
     *  有一个地图点，在多个关键帧上都有它的投影（假如共有n个关键帧可以观测到该地图点），
     * 其中呢，我们设它第一个关键帧上的观测方向向量为：该点坐标指向拍摄该关键帧的相机
     * 光心坐标的单位化，因为总共有n个这样的单位观测方向向量，那么该点的平均观测方向向量为，
     * 取所有单位观测方向向量和的平均数
     * @return  mNormalVector
     * 获得某个地图点的平均观测方向（是一个单位向量）
     */
    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId; ///< Global ID for MapPoint
    static long unsigned int nNextId;
    const long int mnFirstKFid; ///< 创建该MapPoint的关键帧ID
    const long int mnFirstFrame; ///< 创建该MapPoint的帧ID（即每一关键帧有一个帧ID）
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    // TrackLocalMap - SearchByProjection中决定是否对该点进行投影的变量
    // mbTrackInView==false的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    // c 不在当前相机视野中的点（即未通过isInFrustum判断）
    bool mbTrackInView;
    // TrackLocalMap - UpdateLocalPoints中防止将MapPoints重复添加至mvpLocalMapPoints的标记
    long unsigned int mnTrackReferenceForFrame;
    // TrackLocalMap - SearchLocalPoints中决定是否进行isInFrustum判断的变量
    // mnLastFrameSeen==mCurrentFrame.mnId的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;

protected:

    // Position in absolute coordinates
    cv::Mat mWorldPos; ///< MapPoint在世界坐标系下的坐标

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame*,size_t> mObservations; ///< 观测到该MapPoint的KF和该MapPoint在KF中的索引

    // Mean viewing direction
    // 该MapPoint平均观测方向
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    // 每个3D点也有一个descriptor
    // 如果MapPoint与很多帧图像特征点对应（由keyframe来构造时），那么距离其它描述子的平均距离最小的描述子是最佳描述子
    // MapPoint只与一帧的图像特征点对应（由frame来构造时），那么这个特征点的描述子就是该3D点的描述子
    cv::Mat mDescriptor; ///< 通过 ComputeDistinctiveDescriptors() 得到的最优描述子

    // Reference KeyFrame
    KeyFrame* mpRefKF;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapPoint* mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map* mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};
#endif // MAPPOINT_H

