#include "include/mesh.h"

#define Distance_InMesh 4.0f
#define Distance_filter 4.0f


// 两种初始化方式
MyMesh::MyMesh(Map* pMap):mpMap(pMap){}

MyMesh::MyMesh(){}

void MyMesh::Run()
{
    vector<cv::Point3f> vertexs;// 所有顶点

    GetallMappoint(vertexs);

    if(vertexs.size()>100)
        filter_MapPoints(vertexs);

    MakeMesh(vertexs);

    // UpdateVFlist();
}

/**
 * @brief MyMesh::GetallMappoint3
 * @param mappoints 获得地图中所有3D点
 */
void MyMesh::GetallMappoint(vector<cv::Point3f> &mappoints)
{
    vector<MapPoint*> mpmappoint = mpMap->GetAllMapPoints();
    if(mpmappoint.empty())
        return;
    for(size_t i=0; i<mpmappoint.size(); i++)
    {
        MapPoint* pMP = mpmappoint[i];
        if(pMP->isBad())
            continue;
        cv::Point3f p3d;
        cv::Mat point3D = pMP->GetWorldPos();
        p3d.x = point3D.at<float>(0,0);
        p3d.y = point3D.at<float>(0,1);
        p3d.z = point3D.at<float>(0,2);
        mappoints.push_back(p3d);
    }
}

void MyMesh::MakeMesh(vector<cv::Point3f> &mappoints)
{
    MyMesh m;
    float radius = 5.0f;
    float clustering = 0.02f;
    float angle = M_PI/2;
    for(size_t i = 0; i < mappoints.size(); i++)
    {
        float x = mappoints[i].x;
        float y = mappoints[i].y;
        float z = mappoints[i].z;
        vcg::Point3f vertex(x,y,z);
        tri::Allocator<MyMesh>::AddVertex(m, vertex);
    }
    vcg::tri::UpdateBounding<MyMesh>::Box(m);
    vcg::tri::UpdateNormal<MyMesh>::PerFace(m);
    // cout<<"is there a question?"<<endl;
    tri::BallPivoting<MyMesh> pivot(m, radius, clustering, angle);
    // cout<<"no there isn't a question...."<<endl;
    pivot.BuildMesh();
    // cout<<"yes there is a question....."<<endl;

    // remove the unreferenced vertex
    tri::Clean<MyMesh>::RemoveDuplicateVertex(m);
    tri::Clean<MyMesh>::RemoveUnreferencedVertex(m);

    vcg::tri::Allocator<MyMesh>::CompactEveryVector(m);

    MyMesh subM;

    if(mappoints.size()>400)
    {
        tri::SurfaceSampling<MyMesh,tri::TrivialPointerSampler<MyMesh>>::SamplingRandomGenerator().initialize(time(0));
        float perc = 0.04f;

        float radius1 = m.bbox.Diag()*perc;
        tri::SurfaceSampling<MyMesh,tri::MeshSampler<MyMesh>>::PoissonDiskParam pp;
        tri::SurfaceSampling<MyMesh,tri::MeshSampler<MyMesh>>::PoissonDiskParam::Stat pds;
        pp.pds = pds;
        pp.bestSampleChoiceFlag = false;
        tri::MeshSampler<MyMesh> mps(subM);
        tri::SurfaceSampling<MyMesh,tri::MeshSampler<MyMesh>>::PoissonDiskPruning(mps,m,radius1,pp);
        tri::BallPivoting<MyMesh> pivot1(subM,radius,clustering,angle);
        pivot1.BuildMesh();

        // remove the unreferenced vertex
        tri::Clean<MyMesh>::RemoveDuplicateVertex(subM);
        tri::Clean<MyMesh>::RemoveUnreferencedVertex(subM);
    }
    else
        vcg::tri::Append<MyMesh,MyMesh>::MeshCopy(subM,m);

    {
        vector<cv::Point3f> vertexs;
        vertexs.reserve(subM.VN());
        MyMesh::VertexIterator vi;
        MyMesh::VertexPointer vp;
        int idx= 0;
        for(vi=subM.vert.begin();vi!=subM.vert.end(); vi++)
        {
            if(!(*vi).IsD())
            {
                cv::Point3f p1;
                vp=&(*vi);

                p1.x = vp->P()[0];
                p1.y = vp->P()[1];
                p1.z = vp->P()[2];
                vertexs.push_back(p1);
            }
        }

        MyMesh::FaceIterator fi;
        mplist_triangles.clear();
        mplist_triangles.reserve(subM.FN());
        mpMap->clearMeshFace();

        // also use vu as the index of face
        for(fi=subM.face.begin(); fi!=subM.face.end(); fi++)
        {
            cv::Point3f p1,p2,p3;
            if(!(*fi).IsD())
            {
                for(int j=0; j<3; j++)
                {
                    MyMesh::VertexType* pvf = (*fi).V(j);
                    if(j==0)
                    {
                        p1.x = pvf->P().X();
                        p1.y = pvf->P().Y();
                        p1.z = pvf->P().Z();
                    }
                    else if(j==1)
                    {
                        p2.x = pvf->P().X();
                        p2.y = pvf->P().Y();
                        p2.z = pvf->P().Z();
                    }
                    else
                    {
                        p3.x = pvf->P().X();
                        p3.y = pvf->P().Y();
                        p3.z = pvf->P().Z();
                    }
                }
                Triangle* ff = new Triangle(idx,p1,p2,p3);
                mplist_triangles.push_back(ff);
                mpMap->AddMeshFaces(ff);
                idx++;
            }
        }
        mpMapPoints = vertexs;
        // cout<<"Mesh's vertexs: "<< vertexs.size()<<endl;
    }
}

bool MyMesh::backproject2DPoint(const Mat &K, const Mat &Tcw, const cv::Point2f &point2d, cv::Point3f &point3d)
{
    cv::Mat Rcw;
    cv::Mat tcw;
    Rcw=Tcw.rowRange(0,3).colRange(0,3);
    tcw=Tcw.rowRange(0,3).col(3);

    float scale = 8.0f;
    float u = point2d.x;
    float v = point2d.y;

    cv::Mat point2d_vec = cv::Mat::eye(3,1,CV_32F);
    point2d_vec.at<float>(0) = u*scale;
    point2d_vec.at<float>(1) = v*scale;
    point2d_vec.at<float>(2) = scale;

    cv::Mat X_c = K.inv()*point2d_vec;// K^(-1)*S*P(pix)=T*Pw

    cv::Mat X_w = Rcw.inv()*(X_c-tcw);// X_w = Pw

    cv::Mat C_op = cv::Mat(Rcw.inv()).mul(-1)*tcw;// center of projection

    cv::Mat ray = X_w - C_op;
    ray= ray/cv::norm(ray);

    // set ray
    Ray R((cv::Point3f)C_op,(cv::Point3f)ray);

    std::vector<cv::Point3f> intersections_list;

    std::vector<Triangle*> vptri_face=mplist_triangles;

    for(size_t i = 0; i < vptri_face.size(); i++)
    {
        double out;
        if(this->interect_MollerTrumbore(R, vptri_face[i], &out))
        {
            cv::Point3f tmp_pt = R.getP0() + out*R.getP1();
            intersections_list.push_back(tmp_pt);
        }
    }
    if(!intersections_list.empty())
    {
        point3d=get_nearest_3D_point(intersections_list,R.getP0());
        return true;
    }
    else
        return false;

}

void MyMesh::filter_MapPoints(vector<cv::Point3f>& mappoints)
{
    // radius-filter mappoints
    int Th = 3; // the least points of a circle
    float R = Distance_filter; // the radius of circle

    // first point index to second point index ,and the distance*distance
    std::vector<std::pair<std::pair<int,int>,float>> list_index_D;
    int N = mappoints.size();

    for(int i = 0; i < N-1; i++)
        for(int j = i+1; j < N; j++)
        {
            float d = std::sqrt(pow(mappoints[i].x-mappoints[j].x,2)+
                                pow(mappoints[i].y-mappoints[j].y,2)+
                                pow(mappoints[i].z-mappoints[j].z,2));
            list_index_D.push_back(make_pair(make_pair(i,j),d));
        }
    int Nd = list_index_D.size();
    std::vector<int> index(N,0);
    for(int i = 0;i < N; i++)
    {
        int num = 0;
        float D1;
        for(int j = 0; j < Nd; j++)
        {
            if(list_index_D[j].first.first == i || list_index_D[j].first.second == i)
            {
                D1 = list_index_D[j].second;
                if(D1 <= R)
                    num += 1;
            }
            if(num > Th)
            {
                index[i] = 1;
                break;
            }
        }
    }

    // delete isolate point
    int i = 0;
    for(std::vector<cv::Point3f>::iterator it = mappoints.begin(); it != mappoints.end();)
    {
        if(index[i] == 0)
            it = mappoints.erase(it);
        else
            ++it;
        ++i;
    }

    mpMapPoints = mappoints;
}


// 求两向量的外积（叉乘）
cv::Point3f MyMesh::CROSS(const cv::Point3f &V1, const cv::Point3f &V2)
{
    cv::Point3f tmp_p;
    tmp_p.x = V1.y*V2.z - V1.z*V2.y;
    tmp_p.y = V1.z*V2.x - V1.x*V2.z;
    tmp_p.z = V1.x*V2.y - V1.y*V2.x;
    return tmp_p;
}

// 求两向量的内积（点乘）
float MyMesh::DOT(const cv::Point3f &V1, const cv::Point3f &V2)
{
    return V1.x*V2.x + V1.y*V2.y + V1.z*V2.z;
}

cv::Point3f MyMesh::SUB(const cv::Point3f &V1, const cv::Point3f &V2)
{
    cv::Point3f tmp_p;
    tmp_p.x = V1.x - V2.x;
    tmp_p.y = V1.y - V2.y;
    tmp_p.z = V1.z - V2.z;
    return tmp_p;
}

cv::Point3f MyMesh::get_nearest_3D_point(
        const std::vector<cv::Point3f> &points_list,
        const cv::Point3f &origin)
{
    int minidx = 0;
    float mindistance = 50.0;
    for(size_t i=0; i< points_list.size(); i++)
    {
        float distance =std::sqrt(std::pow(points_list[i].x-origin.x,2)+
                                  std::pow(points_list[i].y-origin.y,2)+
                                  std::pow(points_list[i].z-origin.z,2));
        if(distance<mindistance)
        {
            mindistance=distance;
            minidx = i;
        }
    }
    return points_list[minidx];
}

bool MyMesh::interect_MollerTrumbore(Ray &R, Triangle *Tri, double *out)
{
    const double EPSILON = 0.000001;
    cv::Point3f e1,e2;
    cv::Point3f P,Q,T;
    double det, inv_det, u, v;
    double t;

    cv::Point3f V1 = Tri->getV0();
    cv::Point3f V2 = Tri->getV1();
    cv::Point3f V3 = Tri->getV2();

    cv::Point3f O = R.getP0();
    cv::Point3f D = R.getP1();

    e1 = SUB(V2,V1);
    e2 = SUB(V3,V1);

    P = CROSS(D,e2);

    det = DOT(e1,P);
    if(det>-EPSILON && det<EPSILON)
        return false;
    inv_det = 1.0f/det;

    T = SUB(O,V1);

    u = DOT(T,P)*inv_det;

    if(u<0.f || u>1.f)
        return false;

    Q = CROSS(T,e1);

    v = DOT(D,Q)*inv_det;

    if(v<0.f || u+v > 1.f)
        return false;

    t = DOT(e2,Q)*inv_det;

    if(t > EPSILON){
        *out = t;
        return true;
    }

    return false;
}

/**
 * @brief MyMesh::InMesh
 * @param p 输入的3D点
 * @return 如果该点在网格内，或者网格边缘，则返回true，否则返回false
 */
bool MyMesh::InMesh(const cv::Point3f &p)
{
    bool isinmesh = false;
    float thr = Distance_InMesh;
    float midd;
    std::vector<Triangle*> vptri_face = mplist_triangles;
    for(size_t i=0; i<vptri_face.size(); i++)
    {
        Triangle* triface = vptri_face[i];
        cv::Point3f v1,v2,v3;
        v1=triface->getV0();
        v2=triface->getV1();
        v3=triface->getV2();
        float d1,d2,d3;
        d1 = std::sqrt(pow(v1.x-p.x,2)+pow(v1.y-p.y,2)+pow(v1.z-p.z,2));
        d2 = std::sqrt(pow(v2.x-p.x,2)+pow(v2.y-p.y,2)+pow(v2.z-p.z,2));
        d3 = std::sqrt(pow(v3.x-p.x,2)+pow(v3.y-p.y,2)+pow(v3.z-p.z,2));
        midd = (d1+d2+d3)/3.0f;
        if(midd < thr){
            isinmesh = true;
            break;
        }
        else
            isinmesh = false;
    }
    return isinmesh;
}
