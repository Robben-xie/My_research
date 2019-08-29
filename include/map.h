#ifndef MAP_H
#define MAP_H
/* *
 *                 地图由两部分组成：一是，关键帧；
 *            二是，三角化恢复的3D点（又称为地图点）
 * */
#include "mappoint.h"
#include "keyframe.h"
#include "triangle.h"
#include <set>

#include <mutex>

class MapPoint;
class KeyFrame;
class Triangle;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);    /// 添加关键帧
    void AddMapPoint(MapPoint* pMP);   /// 添加地图点（3D点）
    void EraseMapPoint(MapPoint* pMP); /// 擦（删）除地图点
    void AddMeshFaces(Triangle* Tri);
    void EraseKeyFrame(KeyFrame* pKF); /// 擦除关键帧
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);


    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();
    std::vector<Triangle*> GetAllMeshFaces();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clearMeshFace();

    void clear();

    std::vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

protected:
    std::set<MapPoint*> mspMapPoints; ///< MapPoints
    std::set<KeyFrame*> mspKeyFrames; ///< Keyframs
    std::set<Triangle*> mspMeshFaces; ///< TriMeshFaces

    std::vector<MapPoint*> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    std::mutex mMutexMap;
};
#endif // MAP_H

