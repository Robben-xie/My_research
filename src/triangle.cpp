#include <include/triangle.h>
#include <include/mappoint.h>

Triangle::Triangle(int id, cv::Point3f V0, cv::Point3f V1, cv::Point3f V2)
{
    id_ = id;
    V0_ = V0;
    V1_ = V1;
    V2_ = V2;
}

Triangle::~Triangle(){}

Ray::Ray(cv::Point3f P0, cv::Point3f P1)
{
    P0_ = P0;
    P1_ = P1;
}

Ray::~Ray(){}

