#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include<string.h>
#include<malloc.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<termios.h>
#include"math.h"

#include <errno.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>  //for adjust exposure and resolution manually.

#define BUFFER_EXPOSURE_VALUE 150
//#define DEFAULT_FRAME_WIDTH 1280
//#define DEFAULT_FRAME_HEIGHT 720
#define DEFAULT_FRAME_WIDTH 640
#define DEFAULT_FRAME_HEIGHT 480

using namespace cv;
using namespace std;
const int ARROW_AREA_MIN=3000;//variable
const int ARROW_AREA_MAX=600*440;
Mat pro_after;
int cx=640/2;   //To change to calibration parameter.
int cy=480/2;
int Color_detect(Mat frame);

/** @function main */
void set_camera_exposure(int val) {
    int cam_fd;
    if ((cam_fd = open("/dev/video0", O_RDWR)) == -1) {
        cerr << "Camera open error" << endl;
        exit(0);
    }
    struct v4l2_control control_s;
    control_s.id = V4L2_CID_EXPOSURE_AUTO;
    control_s.value = V4L2_EXPOSURE_MANUAL;
    ioctl(cam_fd, VIDIOC_S_CTRL, &control_s);
    control_s.id = V4L2_CID_EXPOSURE_ABSOLUTE;
    control_s.value = val;
    ioctl(cam_fd, VIDIOC_S_CTRL, &control_s);
    close(cam_fd);
}

int main(int argc, char** argv)
{
    Mat src;
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT);
    set_camera_exposure(BUFFER_EXPOSURE_VALUE);

    if(!cap.isOpened()) {
        cout<<"cap open failed..."<<endl;
        exit(0);
    }
    cap.read(src);
    cap.read(src);
    cap.read(src);

    while(cap.read(src))
    {
        imshow("src",src);
        Color_detect(src);
        if('q'==(char)waitKey(7)) break;
    }
    return 0;
}
int Color_detect(Mat frame)
{
    vector<Mat> HSVSplit;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    vector<vector<Point> > contours;
    Mat HSVImage;
    Mat out;
    Mat HGreen;
    Mat SGreen;
    Rect r;
    Rect max_tmp;

    cvtColor(frame, HSVImage, CV_BGR2HSV);
    split(HSVImage,HSVSplit);
    inRange(HSVSplit[0], Scalar(80), Scalar(110), HGreen);
    threshold(HSVSplit[1], SGreen, 80, 255, THRESH_BINARY);
    cv::bitwise_and(HGreen, SGreen, out);
    morphologyEx(out, out, MORPH_OPEN, element);//open operator,remove isolated noise points.
    morphologyEx(out, out, MORPH_CLOSE, element);//close operator,get connected domain.BMC
    Mat solid;
    solid=out.clone();
    findContours(out,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    if( contours.size() == 0 )
    {
        cout<<"no contour..."<<endl;
        return 0;
    }
    Mat result(out.size(),CV_8U,Scalar(0));
    drawContours(result,contours,-1,Scalar(255),2);
    max_tmp=boundingRect(Mat(contours[0]));
    for(int i=1; i<contours.size(); i++)
    {
        r = boundingRect(Mat(contours[i]));
        max_tmp=(r.area()>max_tmp.area())?r:max_tmp;
    }
    if(max_tmp.area()<ARROW_AREA_MIN||max_tmp.area()>ARROW_AREA_MAX)
        return 0;
    Mat pro;
    solid(max_tmp).copyTo(pro);
    pro_after=pro.clone();
    rectangle(result, max_tmp, Scalar(255), 2);
    cout<<"area "<<max_tmp.area()<<endl;
    
    imshow("result",result);
    if('q'==(char)waitKey(7)) exit(0);
    
    Moments mt;
    mt=moments(pro_after,true);
    Point center;
    int diff_x,diff_y;
    center.x=mt.m10/mt.m00+max_tmp.tl().x;
    center.y=mt.m01/mt.m00+max_tmp.tl().y;
    diff_x=center.x-cx;
    diff_y=center.y-cy;
    cout<<"diff x: "<<diff_x<<endl;
    cout<<"diff y: "<<diff_y<<endl;
    return 1;
    //cout<<endl<<"time "<<((double)getTickCount() - t) /(double)getTickFrequency()<<endl;
}
