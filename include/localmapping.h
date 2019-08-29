#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H
#include "keyframe.h"
#include "map.h"
#include "loopclosing.h"
#include "tracking.h"
#include "keyframedatabase.h"
#include "mesh.h"

#include <mutex>

class Tracking;
class LoopClosing;
class Map;
class MyMesh;

class LocalMapping
{
public:

    // 构造函数
    LocalMapping(Map* pMap);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    // 插入关键帧
    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch 线程同步

    void RequestStop();    /// 请求停止运行局部地图
    void RequestReset();  ///  请求重置，即重新初始化等
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    // 返回等待处理的关键帧数量
    int KeyframesInQueue(){
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    void GetcenterAndLength();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);



    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    MyMesh* mesh;

    cv::Mat Centroid;
    float MaxLength;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    // Tracking线程向LocalMapping中插入关键帧是先插入到该队列中
    std::list<KeyFrame*> mlNewKeyFrames; ///< 等待处理的关键帧列表

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
};
#endif // LOCALMAPPING_H

