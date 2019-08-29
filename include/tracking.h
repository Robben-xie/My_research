#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "viewer.h"
#include "framedrawer.h"
#include "map.h"
#include "localmapping.h"
#include "loopclosing.h"
#include "frame.h"
#include "orbvocabulary.h"
#include "keyframedatabase.h"
#include "orbextractor.h"
#include "initializer.h"
#include "mapdrawer.h"
#include "system.h"
#include "mesh.h"

#include <mutex>
#include <thread>

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const std::string &strSettingPath);

    // Preprocess the input and call Track().
    //预处理输入并调用Track（）。
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal length should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const std::string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);


public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    // 初始化时前两帧相关变量
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;// 跟踪初始化时前两帧之间的匹配
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    std::list<cv::Mat> mlRelativeFramePoses;// 位姿
    std::list<KeyFrame*> mlpReferences;// 关键帧
    std::list<double> mlFrameTimes;// 帧数列表
    std::list<bool> mlbLost;// LOST帧标记列表

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;//如果当前地图是无效的并且我们只作用定位，则标记为1

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool TrackWithLastFrame();
    void CreateTwoFrameMapMonocular();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    // orb特征提取器，不管单目还是双目，mpORBextractorLeft都要用到
    // 如果是单目，在初始化的时候使用mpIniORBextractor而不是mpORBextractorLeft，
    // mpIniORBextractor属性中提取的特征点个数是mpORBextractorLeft的两倍
    ORBextractor* mpORBextractorLeft;
    ORBextractor* mpIniORBextractor;

    // Mesh
    MyMesh* Mesh;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    // 单目初始器
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;// 当前关键帧就是参考帧
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;

    // System
    System* mpSystem;

    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;//构建的地图

    //Calibration matrix
    cv::Mat mK;//摄像头内参阵
    cv::Mat mDistCoef;//畸变矩阵
    cv::Mat mTcr;
    cv::Mat mLastFramePos;

    cv::Rect2d bbox;// 选取的目标部分矩形图像

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;//上一关键帧
    Frame mLastFrame;//上一帧
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;// 上一次重定位的那一帧的索引

    //Motion Model 运动模型
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    std::list<MapPoint*> mlpTemporalPoints;
};

#endif // TRACKING_H
