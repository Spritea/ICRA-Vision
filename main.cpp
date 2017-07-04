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

#include <fstream>
//#define _SHOW_PHOTO
#define  _SHOW_POINT_PHOTO
#define _NO_FILTER
//#define _USE_PHOTO
#define _USE_SLR_CAMERA
//#define _SLOW
//#define _OUTPUT_2_TXT
#define BUFFER_EXPOSURE_VALUE 150
//#define DEFAULT_FRAME_WIDTH 1280
//#define DEFAULT_FRAME_HEIGHT 720
#define DEFAULT_FRAME_WIDTH 640
#define DEFAULT_FRAME_HEIGHT 480
using namespace cv;
using namespace std;
const int ARROW_AREA_MIN=3000;//variable
const int ARROW_AREA_MAX=600*440;
int flag=0;
int coincide=0;
Mat pro_after;
Mat cameraMatrix=(Mat_<double>(3,3)<<574.6721,0,335.6625,0,574.6568,223.9030,0,0,1);
//for 640x480.
Mat distCoeffs=(Mat_<double>(1,4)<<0.0202,-0.0189,-0.0036,-0.0030);
// Point3f world_pnt_top(100,60,0);   //unit: mm
// Point3f world_pnt_down(100,114,0);
// Point3f world_pnt_left(83,77,0);
// Point3f world_pnt_right(117,77,0);
Point3f world_pnt_top(0,-40,0);   //unit: mm
Point3f world_pnt_down(0,14,0);
Point3f world_pnt_left(-17,-23,0);
Point3f world_pnt_right(17,-23,0);
Point2f pnt_origin;
int Color_detect(Mat frame);
int Arrow_direction(Mat pro_after);
int Circle_detect(Mat src);
double ty,tz,tx;
void Calcu_attitude(Point3f world_pnt_top,Point3f world_pnt_down,Point3f world_pnt_left,Point3f world_pnt_right,Point2f pnt_top_src,Point2f pnt_down_src,Point2f pnt_left_src,Point2f pnt_right_src);
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
/** @function main */
int main(int argc, char** argv)
{
    int direction=0;//1,2,3,4=arrow,5=circle,6=robo.
    int Iscircle;
    int Isarrow;
    Mat src;
    /// Read the image
#ifdef _USE_PHOTO
    src = imread( argv[1], 1 );
    resize(src,src,Size(src.cols/5,src.rows/5));
    Iscircle=Circle_detect(src);
    if(Iscircle==-1)
    {
        cout<<"image read no data in circle detect..."<<endl;
        return -1;
    }
    if(Iscircle==1)
        direction=5;
    else
    {
        Isarrow=Color_detect(src);
        if(Isarrow==1)
        {
            direction=Arrow_direction(pro_after);
            if(direction==-1)
            {
                cout<<"the arrow position wrong..."<<endl;
                return -1;
            }
        }
        else
            direction=6;
    }
    cout<<"direct: "<<direction<<endl;
#endif

#ifdef _USE_SLR_CAMERA
    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cout<<"no cap! "<<endl;
        return -1;
    }
    cap.set(CV_CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT);
    set_camera_exposure(BUFFER_EXPOSURE_VALUE);

    cap.read(src);
    cap.read(src);
    cap.read(src);
    while(cap.read(src))
    {
//        namedWindow("src");
//        imshow("src",src);
        Iscircle=Circle_detect(src);
        if(Iscircle==-1)
        {
            cout<<"image read no data in circle detect..."<<endl;
            return -1;
        }
        if(Iscircle==1)
            direction=5;
        else
        {
            Isarrow=Color_detect(src);
            if(Isarrow==1)
            {
                direction=Arrow_direction(pro_after);
                if(direction==-1)
                {
                    cout<<"the arrow position wrong..."<<endl;
                    //return -1;
                    continue;
                }
            }
            else
                direction=6;
        }
        cout<<"direct: "<<direction<<endl;
    }
#endif
    return 0;
}

int Arrow_direction(Mat pro_after)
{
    Mat proj;
    bool  Istop_down;
    if(pro_after.rows>pro_after.cols)
        Istop_down=true;
    else
        Istop_down=false;
    if(Istop_down)
        reduce(pro_after,proj,1,CV_REDUCE_AVG,-1);
    else
        reduce(pro_after,proj,0,CV_REDUCE_AVG,-1);
    //string t1 =  type2str( proj.type() );
    //printf("Matrix: %s %dx%d \n", t1.c_str(), proj.cols, proj.rows );
    cout<<"mat size "<<proj.rows<<" "<<proj.cols<<endl;
    int position=0;
    int max=100;
    if(Istop_down)
    {
        for(int row=0; row<proj.rows; ++row)
        {
            const uchar *ptr = proj.ptr(row);
            if((int)ptr[0]>max)
            {
                max=(int)ptr[0];
                position=row;
            }
        }
    }
    else
    {
        const uchar *ptr = proj.ptr(0);
        for(int col=0; col<proj.cols; ++col)
        {
            if((int)ptr[col]>max)
            {
                max=(int)ptr[col];
                position=col;
            }
        }
    }
    cout<<"position "<<position<<endl;
    double scale=(double)position/((proj.rows>proj.cols)?proj.rows:proj.cols);
    cout<<scale<<endl;
    if(scale<0.4&&Istop_down)
    {
        cout<<"direction: top"<<endl;
        return 1;
    }
    else if(scale>0.6&&Istop_down)
    {
        cout<<"direction: down"<<endl;
        return 2;
    }
    else if(scale<0.4&&!Istop_down)
    {
        cout<<"direction: left"<<endl;
        return 3;
    }
    else if(scale>0.6&&!Istop_down)
    {
        cout<<"direction: right"<<endl;
        return 4;
    }
    else
        return -1;
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
    //inRange(HSVSplit[0], Scalar(40), Scalar(80), HGreen);
    //inRange(HSVSplit[0], Scalar(100), Scalar(140), HGreen);
    inRange(HSVSplit[0], Scalar(80), Scalar(110), HGreen);
    //imshow("hgreen",HGreen);
    threshold(HSVSplit[1], SGreen, 80, 255, THRESH_BINARY);
    // imshow("sgreen",SGreen);
    //imshow("h",HSVSplit[0]);
    //imshow("v",HSVSplit[2]);
    cv::bitwise_and(HGreen, SGreen, out);
#ifndef _NO_FILTER
    morphologyEx(out, out, MORPH_OPEN, element);//open operator,remove isolated noise points.
    morphologyEx(out, out, MORPH_CLOSE, element);//close operator,get connected domain.BMC
#endif

#ifdef _SHOW_PHOTO
    imshow("before contour",out);
#endif
    Mat solid;
    solid=out.clone();
#ifdef _SHOW_PHOTO
    imshow("solid",solid);
    //cout<<"debug"<<endl;
#endif
    findContours(out,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    if( contours.size() == 0 )
    {
        //write(fd,lost,18);
        //continue;
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
#ifdef _SHOW_PHOTO
    imshow("pro",pro);
#endif
    Point2f pnt_top_default((max_tmp.tl().x+max_tmp.br().x)/2,max_tmp.tl().y);
    Point2f pnt_down_default(pnt_top_default.x,max_tmp.br().y);
    Point2f pnt_left_default(max_tmp.tl().x,max_tmp.tl().y+17.0/54*max_tmp.height);
    Point2f pnt_right_default(max_tmp.br().x,pnt_left_default.y);

    Mat top=pro.rowRange(0,1).clone();
    Mat down=pro.rowRange(pro.rows-1,pro.rows).clone();
    Mat left=pro.colRange(0,1).clone();
    Mat right=pro.colRange(pro.cols-1,pro.cols).clone();
    //cout<<"top: "<<top<<endl;
    //cout<<"down: "<<down<<endl;
    //cout<<"left: "<<left<<endl;
    //cout<<"right: "<<right<<endl;
    vector<Point> loca_top;
    //vector<Point> loca_down;   //the middle of down line.
    vector<Point> loca_left;
    vector<Point> loca_right;
    findNonZero(top,loca_top);
    //findNonZero(down,loca_down);
    findNonZero(left,loca_left);
    findNonZero(right,loca_right);
    Point pnt_top;
    //Point pnt_down;
    Point pnt_left;
    Point pnt_right;
    int d1,d2,d3;
    float min_top=abs(loca_top.at(0).x+max_tmp.tl().x-pnt_top_default.x);
    for(int i=0; i<loca_top.size(); i++)
    {
        if(abs(loca_top.at(i).x+max_tmp.tl().x-pnt_top_default.x)<=min_top)
        {
            min_top=abs(loca_top.at(i).x+max_tmp.tl().x-pnt_top_default.x);
            d1=i;
        }
    }
    pnt_top.x=loca_top.at(d1).x;
    pnt_top.y=loca_top.at(d1).y;

    float min_left=abs(loca_left.at(0).y+max_tmp.tl().y-pnt_left_default.y);
    for(int i=0; i<loca_left.size(); i++)
    {
        if(abs(loca_left.at(i).y+max_tmp.tl().y-pnt_left_default.y)<=min_left)
        {
            min_left=abs(loca_left.at(i).y+max_tmp.tl().y-pnt_left_default.y);
            d2=i;
        }
    }
    pnt_left.x=loca_left.at(d2).x;
    pnt_left.y=loca_left.at(d2).y;

    float min_right=abs(loca_right.at(0).y+max_tmp.tl().y-pnt_right_default.y);
    for(int i=0; i<loca_right.size(); i++)
    {
        if(abs(loca_right.at(i).y+max_tmp.tl().y-pnt_right_default.y)<=min_right)
        {
            min_right=abs(loca_right.at(i).y+max_tmp.tl().y-pnt_right_default.y);
            d3=i;
        }
    }
    pnt_right.x=loca_right.at(d3).x;
    pnt_right.y=loca_right.at(d3).y;
    cout<<"d1 "<<d1<<endl;
    cout<<"d2 "<<d2<<endl;
    cout<<"d3 "<<d3<<endl;
//     Point pnt_top=loca_top.at(0);
//     //Point  pnt_down=loca_down.at(0);
//     Point pnt_left=loca_left.at(0);
//     Point pnt_right=loca_right.at(0);
    Point2f pnt_top_src(pnt_top.x+max_tmp.tl().x,pnt_top.y+max_tmp.tl().y);
    Point2f pnt_down_src(pnt_top_src.x,max_tmp.br().y);
    Point2f pnt_left_src(pnt_left.x+max_tmp.tl().x,pnt_left.y+max_tmp.tl().y);
    Point2f pnt_right_src(pnt_right.x+pro.cols-1+max_tmp.tl().x,pnt_right.y+max_tmp.tl().y);
    Point2f pnt_origin_src(pnt_top_src.x,40/54.0*max_tmp.height+pnt_top_src.y);
//     cout<<"top: "<<pnt_top<<endl;
//     cout<<"down: "<<pnt_down<<endl;
//     cout<<"left: "<<pnt_left<<endl;
//     cout<<"right: "<<pnt_right<<endl;
    circle( frame, pnt_top_src, 5, Scalar(0,0,255), -1, 8, 0 );
    circle( frame, pnt_down_src, 5, Scalar(0,0,255), -1, 8, 0 );
    circle( frame, pnt_left_src, 5, Scalar(0,0,255), -1, 8, 0 );
    circle( frame, pnt_right_src, 5, Scalar(0,0,255), -1, 8, 0 );
    circle( frame, pnt_origin_src, 5, Scalar(0,255,255), -1, 8, 0 );
// #ifdef _SHOW_POINT_PHOTO
//     imshow("point",frame);
// #endif
    //cout<<"top_src: "<<pnt_top_src<<endl;
    cout<<"origin_src"<<pnt_origin_src<<endl;
    pnt_origin.x=pnt_origin_src.x;
    pnt_origin.y=pnt_origin_src.y;
    coincide=0;
    if(abs(pnt_origin_src.x-335.6625)<3&&abs(pnt_origin_src.y-223.9030)>=3)
        coincide=1;
    else if(abs(pnt_origin_src.x-335.6625)>=3&&abs(pnt_origin_src.y-223.9030)<3)
        coincide=2;
    else if(abs(pnt_origin_src.x-335.6625)<3&&abs(pnt_origin_src.y-223.9030)<3)
        coincide=3;
    pro_after=pro.clone();
    //string ty =  type2str( pro_after.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), pro_after.cols, pro_after.rows );
    rectangle(result, max_tmp, Scalar(255), 2);
    cout<<"area "<<max_tmp.area()<<endl;
//    Calcu_attitude(world_pnt_top,world_pnt_down,world_pnt_left,world_pnt_right,pnt_top_src,pnt_down_src,pnt_left_src,pnt_right_src);
    Calcu_attitude(world_pnt_top,world_pnt_down,world_pnt_left,world_pnt_right,pnt_top_default,pnt_down_default,pnt_left_default,pnt_right_default);
#ifdef _SHOW_POINT_PHOTO
    char str_y[20];
    char str_z[20];
    char str_x[20];
    sprintf(str_y,"%lf",ty);
    sprintf(str_z,"%lf",tz);
    sprintf(str_x,"%lf",tx);
    string pre_str_y="thetay: ";
    string pre_str_z="thetaz: ";
    string pre_str_x="thetax: ";
    string full_y=pre_str_y+str_y;
    string full_z=pre_str_z+str_z;
    string full_x=pre_str_x+str_x;
    putText(frame,full_y,Point(30,30),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2);
    putText(frame,full_z,Point(30,70),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,0),2);
    putText(frame,full_x,Point(30,100),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,0),2);
#ifdef _SLOW
    usleep(5*1000*100);
#endif
    imshow("point",frame);
    imshow("result",result);
    if('q'==(char)waitKey(5))
        exit(0);
#endif

    //cout<<endl<<"time "<<((double)getTickCount() - t) /(double)getTickFrequency()<<endl;
    return 1;
}

int Circle_detect(Mat src)
{
    Mat src_gray;
    if( !src.data )
    {
        return -1;
    }
    /// Convert it to gray
    cvtColor( src, src_gray, CV_BGR2GRAY );
    /// Reduce the noise so we avoid false circle detection
    //medianBlur(src_gray, src_gray, 5);
    GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
#ifdef _SHOW_PHOTO
    imshow("gaussian",src_gray);
#endif
    vector<Vec3f> circles;

    /// Apply the Hough Transform to find the circles
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/4, 100, 30, 15, 30 );
    //mainly depends 100,30 the 2 parameters.BMC
    //HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, 10,100, 30, 15, 30);
    // change the last two parameters
    // (min_radius & max_radius) to detect larger circles
    cout<<"circles num: "<<circles.size()<<endl;
    if(circles.size()>=2)   //circle num >=2.BMC
    {
        cout<<"hhh "<<endl;
        /// Draw the circles detected
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }
        /// Show your results
#ifdef _SHOW_PHOTO
        namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
        imshow( "Hough Circle Transform Demo", src );
        waitKey(0);
#endif
        return 1;
    }
    else
        return 0;
}
void Calcu_attitude(Point3f world_pnt_top,Point3f world_pnt_down,Point3f world_pnt_left,Point3f world_pnt_right,Point2f pnt_top_src,Point2f pnt_down_src,Point2f pnt_left_src,Point2f pnt_right_src)
{
    vector<Point3f> Points3D;
    vector<Point2f> Points2D;
    // Vec3d  rvec,tvec;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    Points3D.push_back(world_pnt_top);
    Points3D.push_back(world_pnt_down);
    Points3D.push_back(world_pnt_left);
    Points3D.push_back(world_pnt_right);
    Points2D.push_back(pnt_top_src);
    Points2D.push_back(pnt_down_src);
    Points2D.push_back(pnt_left_src);
    Points2D.push_back(pnt_right_src);
    solvePnP(Points3D,Points2D,cameraMatrix,distCoeffs,rvec,tvec);
    //Mat rotM(3,3,CV_64FC1);
    double rm[9];
    cv::Mat rotM(3, 3, CV_64FC1, rm);
    Rodrigues(rvec,rotM);
    cout<<"tvec: "<<tvec<<endl;
    double r11 = rotM.ptr<double>(0)[0];
//     cout<<"r11: "<<r11<<endl;
//     cout<<"rotM 0,0: "<<rotM.at<double>(0,0)<<endl;
//     cout<<"rm: "<<rm<<endl;
//     cout<<"rotM: "<<rotM<<endl;
    double r12 = rotM.ptr<double>(0)[1];
    double r13 = rotM.ptr<double>(0)[2];
    double r21 = rotM.ptr<double>(1)[0];
    double r22 = rotM.ptr<double>(1)[1];
    double r23 = rotM.ptr<double>(1)[2];
    double r31 = rotM.ptr<double>(2)[0];
    double r32 = rotM.ptr<double>(2)[1];
    double r33 = rotM.ptr<double>(2)[2];

    double thetaz = atan2(r21, r11) / CV_PI * 180;
    double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / CV_PI * 180;
    double thetax = atan2(r32, r33) / CV_PI * 180;
    ty=thetay;
    tz=thetaz;
    tx=thetax;
    cout<<"thetay: "<<thetay<<endl;
    cout<<"thetaz: "<<thetaz<<endl;
    cout<<"thetax: "<<thetax<<endl;
    //double heading=asin(rotM.at<float>(2,0)); //y -PI/2~PI/2
    //double heading=atan2(-rotM.at<float>(2,0),sqrt(pow(rotM.at<float>(2,1),2)+pow(rotM.at<float>(2,2),2)))/CV_PI*180; //y
//     double attitude=atan2(rotM.at<float>(1,0),rotM.at<float>(0,0))/CV_PI*180; //-PI~PI z
//     double bank=atan2(rotM.at<float>(2,1),rotM.at<float>(2,2))/CV_PI*180; //x
//     cout<<"heading: "<<heading<<endl;   //-PI~PI.
//     cout<<"attitude: "<<attitude<<endl;
//     cout<<"bank: "<<bank<<endl;
#ifdef _OUTPUT_2_TXT
    if(coincide>0)
    {
        ofstream out("out.txt",ios::app);
        //ofstream out("out.txt");
        cout<<"out.txt OK ! "<<endl;
        char x[10];
        char y[10];
        char Tx[20];
        char Ty[20];
        char type[2];
        sprintf(x,"%f",pnt_origin.x);
        sprintf(y,"%f",pnt_origin.y);
        sprintf(Tx,"%lf",tvec[0]);
        sprintf(Ty,"%lf",tvec[1]);
        sprintf(type,"%d",coincide);
        if(out.is_open())
        {
            out<<"pnt_origin x: "<<x<<'\n';
            out<<"pnt_origin y: "<<y<<'\n';
            out<<"Tx: "<<Tx<<'\n';
            out<<"Ty: "<<Ty<<'\n';
            out<<"Type: "<<type<<'\n';
            out.close();
        }
    }
#endif

}



























