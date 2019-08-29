#include "include/system.h"
#include "include/converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>		// std::setprecision

bool has_suffix(const std::string &str, const std::string &suffix)
{
    //若str和suffix两个字符串存在包含关系，则返回值必不等于npos，否则等于
    //在str中从第二个参数的位置开始寻找，是否包含suffix字符串，
    std::size_t index = str.find(suffix, str.size() - suffix.size());
    return (index != std::string::npos);
}


System::System(const string &strVocFile, const string &strSettingsFile, const bool bUseViewer):
    mbReset(false),mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false)
{
    //Check settings file
    //这里是对相机的参数文件进行读取
    //深度阈值，对应ORB Extractor的参数设定，还有Viewer线程的参数设定，XXX.yaml这种文件类型
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    //是否打开文件成功
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    //下载对应的词袋模型，对应的是.txt文件类型
    cout << "Loading ORB Vocabulary. This could take a while..." << endl;

    //开辟新的内存
    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = false; // chose loading method based on file extension
    if (has_suffix(strVocFile, ".txt")){
        //将文件载入变量中
        bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    }
    else if(has_suffix(strVocFile, ".bin")){
        bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
    }
    else
      bVocLoad = false;
    //如果文件载入错误
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Failed to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //开辟新的内存，存储关键帧数据
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    // 创建地图
    mpMap = new Map();

    // 创建观测者,两个窗口
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);//第二个参数是相机参数文件

    // 初始化跟踪线程，它将存在于执行的主线程中，即调用此构造函数的线程
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile);

    // 初始化局部地图线程并启动
    mpLocalMapper = new LocalMapping(mpMap);//局部地图
    mptLocalMapping = new thread(&LocalMapping::Run,mpLocalMapper);//局部地图线程

    //初始化回环检测线程并启动
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary);
    mptLoopClosing = new thread(&LoopClosing::Run, mpLoopCloser);

    //初始化可视化线程并启动
    mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);//初始化观测者
    if(bUseViewer)
        mptViewer = new thread(&Viewer::Run, mpViewer);//开辟线程并启动

    mpTracker->SetViewer(mpViewer);

    // 在线程之间设置指针,各线程之间的联系
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

//单目
cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    //第1步:确定模型改变
    // Check mode change
    {
        // a:上锁
        unique_lock<mutex> lock(mMutexMode);
        // b：激活定位模式
        if(mbActivateLocalizationMode)
        {
            // b1:局部地图申请停止
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            // b2:判断是否有效停止，若未停止则延时至少1ms
            while(!mpLocalMapper->isStopped())
            {
                //usleep(1000);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            mpTracker->InformOnlyTracking(true);// 定位时，只跟踪
            mbActivateLocalizationMode = false;// 防止重复执行
        }
        // c:停用定位模式
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;// 防止重复执行
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    return mpTracker->GrabImageMonocular(im,timestamp);
}
//激活定位模型
void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}
//停用定位模型
void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}
//系统重置
void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}
//关闭整个系统
void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Wait until all thread have effectively stopped
    // 直到所有线程都停止
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        //usleep(5000);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if(mpViewer)
        pangolin::BindToContext("SFM: Map Viewer");
}

void System::SaveKeyFrameTrajectoryTUM(const std::string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveMap(const string &filename)
{
    cout<< endl << "Saving All Mappoints to file " << filename <<endl;
    vector<MapPoint*> vpMP = mpMap->GetAllMapPoints();
    int Nv = vpMP.size();
    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    f << "ply" <<endl;
    f << "format ascii 1.0" << endl;
    f << "comment made by anonymous" << endl;
    f << "comment this file is a model" << endl;
    f << "element vertex "+to_string(Nv) << endl;
    f << "property float32 x" << endl;
    f << "property float32 y" << endl;
    f << "property float32 z" << endl;
    f << "element face 0" << endl;
    f << "property list uint8 int32 vertex_index" << endl;
    f << "end_header" << endl;
    for(size_t i = 0; i < vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;

        cv::Mat point3D = pMP->GetWorldPos();
        f << point3D.at<float>(0) << " " <<point3D.at<float>(1) << " " << point3D.at<float>(2) << endl;
    }
    f.close();
    cout<<endl<< "Map have been saved!"<<endl;
}

