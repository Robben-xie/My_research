#include "include/map.h"

#include<mutex>


Map::Map():mnMaxKFid(0)
{
}

/**
 * @brief Insert KeyFrame in the map
 * @param pKF KeyFrame
 */
void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

/**
 * @brief Insert MapPoint in the map
 * @param pMP MapPoint
 */
void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::AddMeshFaces(Triangle *Tri)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMeshFaces.insert(Tri);
}

/**
 * @brief Erase MapPoint from the map
 * @param pMP MapPoint
 */
void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

/**
 * @brief Erase KeyFrame from the map
 * @param pKF KeyFrame
 */
void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::clearMeshFace()
{
    for(set<Triangle*>::iterator sit=mspMeshFaces.begin(), send=mspMeshFaces.end(); sit!=send; sit++)
        delete *sit;
    mspMeshFaces.clear();
}

/**
 * @brief 设置参考MapPoints，将用于DrawMapPoints函数画图
 * @param vpMPs Local MapPoints
 */
void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

//void Map::UpdateMesh(MyMesh* mesh)
//{
//    unique_lock<mutex> lock(mMutexMap);
//    mmesh = mesh;
//}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

vector<Triangle*> Map::GetAllMeshFaces()
{
    unique_lock<mutex> lock(mMutexMap);
    return std::vector<Triangle*>(mspMeshFaces.begin(),mspMeshFaces.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()// 返回关键帧队列的大小
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;
    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;;
    for(set<Triangle*>::iterator sit=mspMeshFaces.begin(), send=mspMeshFaces.end(); sit!=send; sit++)
        delete *sit;
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mspMeshFaces.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}
/*
void Map::Save ( const string& filename )
 {
     //输出地图保存的信息
     cerr<<"Map Saving to "<<filename <<endl;
     ofstream f;
     f.open(filename.c_str(), ios_base::out|ios::binary);
     //输出地图点的数目
     cerr << "The number of MapPoints is :"<<mspMapPoints.size()<<endl;

     //地图点的数目
     unsigned long int nMapPoints = mspMapPoints.size();
     f.write((char*)&nMapPoints, sizeof(nMapPoints) );
     //依次保存MapPoints
     for ( auto mp: mspMapPoints )
     //调用SaveMapPoint函数
         SaveMapPoint( f, mp );

     //获取每一个MapPoints的索引值，即从0开始计数，初始化了mmpnMapPointsIdx
     GetMapPointsIdx();

    //输出关键帧的数量
     cerr <<"The number of KeyFrames:"<<mspKeyFrames.size()<<endl;
     //关键帧的数目
     unsigned long int nKeyFrames = mspKeyFrames.size();
     f.write((char*)&nKeyFrames, sizeof(nKeyFrames));

     //依次保存关键帧KeyFrames
     for ( auto kf: mspKeyFrames )
         SaveKeyFrame( f, kf );

     for (auto kf:mspKeyFrames )
     {
         //获得当前关键帧的父节点，并保存父节点的ID
         KeyFrame* parent = kf->GetParent();
         unsigned long int parent_id = ULONG_MAX;
         if ( parent )
             parent_id = parent->mnId;
         f.write((char*)&parent_id, sizeof(parent_id));
         //获得当前关键帧的关联关键帧的大小，并依次保存每一个关联关键帧的ID和weight；
         unsigned long int nb_con = kf->GetConnectedKeyFrames().size();
         f.write((char*)&nb_con, sizeof(nb_con));
         for ( auto ckf: kf->GetConnectedKeyFrames())
         {
             int weight = kf->GetWeight(ckf);
             f.write((char*)&ckf->mnId, sizeof(ckf->mnId));
             f.write((char*)&weight, sizeof(weight));
         }
     }

     f.close();
     cerr<<"Map Saving Finished!"<<endl;
 }
*/
