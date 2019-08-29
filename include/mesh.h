#ifndef MESH_H
#define MESH_H
#include <opencv2/core/core.hpp>
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/smooth.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/flag.h>
#include <vcg/complex/algorithms/create/ball_pivoting.h>
#include <vcg/complex/algorithms/point_sampling.h>
#include <vcg/complex/algorithms/clustering.h>
#include <mutex>

#include "map.h"
#include "triangle.h"

class MyFace;
class MyVertex;

class Triangle;
class Map;

using namespace vcg;
struct MyUsedTypes:public UsedTypes<Use<MyVertex>::AsVertexType,
        Use<MyFace>::AsFaceType>{};

class MyVertex:public Vertex<MyUsedTypes,vertex::Coord3f,vertex::Normal3f,
        vertex::BitFlags,vertex::Mark>{};

class MyFace:public Face<MyUsedTypes,face::VertexRef,
        face::Normal3f,face::BitFlags>{};

class MyMesh:public vcg::tri::TriMesh<vector<MyVertex>, vector<MyFace>>
{
public:
    MyMesh(Map* pMap);
    MyMesh();

    void GetallMappoint(vector<cv::Point3f> &mappoints);

    bool backproject2DPoint(const cv::Mat &K, const cv::Mat &Tcw, const cv::Point2f &point2d, cv::Point3f &point3d);

    bool interect_MollerTrumbore(Ray &R, Triangle *Tri, double *out);

    void filter_MapPoints(vector<cv::Point3f>& mappoints);

    bool InMesh(const cv::Point3f &p);

    void MakeMesh(vector<cv::Point3f> &mappoints);


    // void UpdateVFlist();

    // main function
    void Run();

    // 这里定义了2个向量的运算函数（叉乘，点乘，相加减）

    float DOT(const cv::Point3f &V1, const cv::Point3f &V2);

    cv::Point3f SUB(const cv::Point3f &V1, const cv::Point3f &V2);

    cv::Point3f get_nearest_3D_point(const std::vector<cv::Point3f> &points_list, const cv::Point3f &origin);

    cv::Point3f CROSS(const cv::Point3f &V1, const cv::Point3f &V2);


protected:

    Map* mpMap;

    // MyMesh* mmMesh;

    std::vector<Triangle*> mplist_triangles;
    std::vector<cv::Point3f> mpMapPoints;
};
#endif // MESH_H
