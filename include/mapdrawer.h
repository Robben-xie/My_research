#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include "map.h"
#include "mappoint.h"
#include "keyframe.h"
#include <pangolin/pangolin.h>

#include <mutex>

class MapDrawer
{
public:
    MapDrawer(Map* pMap, const std::string &strSettingPath);

    Map* mpMap;

    void DrawMapPoints(); // 画地图点
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph); // 画关键帧
    void DrawMyMesh(); // 画网格
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc); // 画当前相机位置
    void SetCurrentCameraPose(const cv::Mat &Tcw); // 设置当前相机的位姿
    void SetReferenceKeyFrame(KeyFrame *pKF); // 获得参考关键帧

    // 获取当前相机外参矩阵，只不过矩阵类型要变换
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

private:

    // OpenGL画图的一些参数
    float mKeyFrameSize; // 关键帧在图上的大小
    float mKeyFrameLineWidth; // 表示关键帧的线宽
    float mGraphLineWidth; // 表示位姿图之间连线的线宽
    float mPointSize; // 地图点大小
    float mCameraSize; // 相机的大小
    float mCameraLineWidth; // 表示相机的线宽

    cv::Mat mCameraPose; // 相机的位姿

    std::mutex mMutexCamera;
};

#endif // MAPDRAWER_H
