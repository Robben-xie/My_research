#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "include/system.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if(argc != 4){
        cout<<" Usage:  ./motion_struct image_path ORBDBoW3Path Parampath"<<endl;
        return false;
    }

    VideoCapture video(argv[1]);

    if(!video.isOpened())
    {
        cout<<"Could not read video file"<<endl;
        return 1;
    }

    System SFM(argv[2], argv[3],true);// 实例化:训练好的词库，相机参数和显示属性参数

    vector<double> vTimesTrack;
    vTimesTrack.reserve(1500);

    cout<<"/******Start processing sequence******/"<<endl;

    Mat frame;
    int nImages = 0;
    double cur_nImages = 0;

    time_t start,end;
    double sec,fps,totaltime;
    time(&start);
    while(video.read(frame))
    {
        if(frame.empty())
        {
            cerr<<endl<<"Failed to load image"<<endl;
            return 1;
        }
        if(nImages==0 || nImages%1==0)
        {
            // 计时，其实不用定义直接用else下的语句就，然后求duration
            #ifdef COMPILEDWITHC11
                    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            #else
                    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
            #endif
//                    cout<<"/*****currentframe's Id = "<<cur_nImages<<"\t*******/"<<endl;

                    // Pass the image to the SLAM system
                    SFM.TrackMonocular(frame,cur_nImages);

            #ifdef COMPILEDWITHC11
                    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            #else
                    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
            #endif
                    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

                    vTimesTrack[cur_nImages]=float(ttrack);
                    totaltime = totaltime + ttrack;
                    cur_nImages++;
        }
        nImages++;
    }
    time(&end);
    sec = difftime(end,start);

    fps = (cur_nImages+1)/sec;
    cout<<"/********fps**********/"<<fps<<endl;

    SFM.Shutdown();

    sort(vTimesTrack.begin(),vTimesTrack.end());

    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[(cur_nImages+1)/2] << endl;
    cout << "mean tracking time: " << double(totaltime/(cur_nImages+1)) << endl;
    cout << "number pictures have been dealed: "<< cur_nImages<<endl;

    // Save camera trajectory
    SFM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SFM.SaveMap("objectmesh.ply");

    return 0;
}
