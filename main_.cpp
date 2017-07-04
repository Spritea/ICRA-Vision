#include "opencv2/opencv.hpp"
#include <iostream>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<termios.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>  //for adjust exposure and resolution manually.
#include <math.h>
using namespace std;
using namespace cv;
#define BUFFER_EXPOSURE_VALUE 150
//#define DEFAULT_FRAME_WIDTH 1280
//#define DEFAULT_FRAME_HEIGHT 720
#define DEFAULT_FRAME_WIDTH 640
#define DEFAULT_FRAME_HEIGHT 480

Mat getHistImg(const MatND& hist);
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

int main()
{
    cout<<"arccos 0.3: "<<acos(0.3)/CV_PI*180<<endl;
    VideoCapture capture;
	capture.open(0);
    
	if (!capture.isOpened())
	{
		cout << " failed to open cap!" << endl;
		return 1;
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT);
    set_camera_exposure(BUFFER_EXPOSURE_VALUE);
    Mat frame;
    Mat grey;
    capture.read(frame);
    capture.read(frame);
    capture.read(frame);
    while(capture.read(frame))
    {
        capture>>frame;
        cvtColor(frame,grey,CV_BGR2GRAY);
        const int channels[1]={0};
        const int histSize[1]={256};
        float hranges[2]={0,255};
        const float* ranges[1]={hranges};
        MatND hist;
        calcHist(&grey,1,channels,Mat(),hist,1,histSize,ranges);
        //cout<<"hist: "<<hist<<endl;
        float sum=0;
        for(int i=0;i<60;i++)
        {
            sum+=hist.at<float>(i);
        }
        cout<<"sum: "<<sum<<endl;
        cout<<"portion: "<<sum/(640*480)<<endl;
        getHistImg(hist);
    }
        return 0;
}
Mat getHistImg(const MatND& hist)
{
    double maxVal=0;
    double minVal=0;
 
    //找到直方图中的最大值和最小值
    minMaxLoc(hist,&minVal,&maxVal,0,0);
    int histSize=hist.rows;
    Mat histImg(histSize,histSize,CV_8U,Scalar(255));
    // 设置最大峰值为图像高度的90%
    int hpt=static_cast<int>(0.9*histSize);
 
    for(int h=0;h<histSize;h++)
    {
        float binVal=hist.at<float>(h);
        int intensity=static_cast<int>(binVal*hpt/maxVal);
        line(histImg,Point(h,histSize),Point(h,histSize-intensity),Scalar::all(0));
    }
    imshow("hist",histImg);
    if('q'==(char)waitKey(5))
        exit(0);
    return histImg;
}
