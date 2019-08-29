#ifndef SYSTEM_H
#define SYSTEM_H
#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "include/tracking.h"
#include "include/framedrawer.h"
#include "include/mapdrawer.h"
#include "include/map.h"
#include "include/localmapping.h"
#include "include/loopclosing.h"
#include "include/keyframedatabase.h"
#include "include/orbvocabulary.h"
#include "include/viewer.h"

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
public:

    //第一个参数应该是ORBvoc，第二个是相机内参畸变矩阵等参数文件，第三个是传感器类型（单目，双目，激光等)
    // Initialize the SFM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const std::string &strVocFile, const std::string &strSettingsFile, const bool bUseViewer = true);

    // Proccess the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);

    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Reset the system (clear map)
    void Reset();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();


    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const std::string &filename);


    // TODO: Save/Load functions
    void SaveMap(const string &filename);

private:

    // ORB vocabulary used for place recognition and feature matching.
    // 用于场所识别和特征匹配的ORB词汇表。
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    // 用于位置识别的关键帧数据库（重定位和循环检测）。
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //存储指向所有关键帧和地图点指针的地图结构。
    Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    // 跟踪，接收一帧并且计算关联的相机位姿
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // 决定何时插入关键帧，创建新的地图点并且如果跟踪失败决定重定位
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // 局部地图，它管理局部地图并且执行局部BA优化
    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    // 回环检测，它会搜索每个新关键帧的循环，
    // 如果存在循环，则之后执行位姿图优化和完整BA优化（在新线程中）。
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    // 观测者绘制地图和当前相机位姿
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    //系统线程：局部地图，回环检测，查看器。跟踪线程“进程”在创建System对象的主执行线程中。
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    //重置标志
    std::mutex mMutexReset;
    bool mbReset;

    // Change mode flags
    // 改变模型标志
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;  //激活定位模式
    bool mbDeactivateLocalizationMode;//停用定位模式
};

#endif // SYSTEM_H
