#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <opencv2/core/core.hpp>

using namespace std;

/**
 * @brief The Triangle class 三角面片类
 */
class Triangle
{
public:
    explicit Triangle(int id, cv::Point3f V0, cv::Point3f V1, cv::Point3f V2);
    virtual ~Triangle();

    cv::Point3f getV0() const {return V0_;}
    cv::Point3f getV1() const {return V1_;}
    cv::Point3f getV2() const {return V2_;}

private:
    // 三角面片的索引
    int id_;

    // 三角面片的三个顶点
    cv::Point3f V0_, V1_, V2_;
};

/**
 * @brief The Ray class 射线类
 */
class Ray
{
public:
    explicit Ray(cv::Point3f P0, cv::Point3f P1);
    virtual ~Ray();

    cv::Point3f getP0(){return P0_;}
    cv::Point3f getP1(){return P1_;}

private:
    // 射线的两端点
    cv::Point3f P0_, P1_;
};
#endif // TRIANGLE_H
