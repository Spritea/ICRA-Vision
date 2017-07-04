// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image
//the approx start from top-left,then clockwise order

#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>
#include <string.h>
#include <algorithm>

#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<termios.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>  //for adjust exposure and resolution manually.
using namespace cv;
using namespace std;
#define BUFFER_EXPOSURE_VALUE 150
//#define DEFAULT_FRAME_WIDTH 1280
//#define DEFAULT_FRAME_HEIGHT 720
#define DEFAULT_FRAME_WIDTH 640
#define DEFAULT_FRAME_HEIGHT 480

int thresh = 50, N =1;
const char* wndname = "Square Detection Demo";

int grey_thresh=60;
int counts=0;
double ry,rz,rx;
double tx,ty,tz;
Mat cameraMatrix=(Mat_<double>(3,3)<<575.1391,0,342.8161,0,574.4738,232.1716,0,0,1);
//for 640x480.
Mat distCoeffs=(Mat_<double>(1,4)<<0.0371,-0.0227,-0.0025,-0.0010);
Point3f world_pnt_tl(-65,-85,0);   //unit: mm
Point3f world_pnt_tr(65,-85,0);
Point3f world_pnt_br(65,85,0);
Point3f world_pnt_bl(-65,85,0);

void Calcu_attitude(Point3f world_pnt_tl,Point3f world_pnt_tr,Point3f world_pnt_br,Point3f world_pnt_bl,Point2f pnt_tl_src,Point2f pnt_tr_src,Point2f pnt_br_src,Point2f pnt_bl_src);
int Color_judge(Mat src,int area);
class MyPoint
{
public:
    MyPoint(Point pnt)
    {
        x=pnt.x;
        y=pnt.y;
    };
    int x;
    int y;
    bool operator<(const MyPoint&p)const
    {
        return x<p.x;
    }
};

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

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( Mat src,const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // tr-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;
    counts=0;
    cout<<"****new****"<<endl<<endl;
    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading

                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
             Canny(gray0, gray, 50, 200, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                //dilate(gray, gray, Mat(), Point(-1,-1));

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
            vector<Point> approx;
            
            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
				if (contours[i].size()<50||contours[i].size()>400)
				{
					continue;
				}
				
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                    {
                        counts++;
                        Point tl,tr,br,bl;
                        vector<Point> rect=approx;
                        MyPoint p1(rect[0]);
                        MyPoint p2(rect[1]);
                        MyPoint p3(rect[2]);
                        MyPoint p4(rect[3]);
                        vector<MyPoint> my_rect;
                        my_rect.push_back(p1);
                        my_rect.push_back(p2);
                        my_rect.push_back(p3);
                        my_rect.push_back(p4);
                        sort(my_rect.begin(),my_rect.end());
                        if(my_rect[0].y<my_rect[1].y)
                        {
                            tl.x=my_rect[0].x;
                            tl.y=my_rect[0].y;
                            bl.x=my_rect[1].x;
                            bl.y=my_rect[1].y;
                        }
                        else
                        {
                            tl.x=my_rect[1].x;
                            tl.y=my_rect[1].y;
                            bl.x=my_rect[0].x;
                            bl.y=my_rect[0].y;
                        }
                        if(my_rect[2].y<my_rect[3].y)
                        {
                            tr.x=my_rect[2].x;
                            tr.y=my_rect[2].y;
                            br.x=my_rect[3].x;
                            br.y=my_rect[3].y;
                        }
                        else
                        {
                            tr.x=my_rect[3].x;
                            tr.y=my_rect[3].y;
                            br.x=my_rect[2].x;
                            br.y=my_rect[2].y;
                        }

                        cout<<"tl x: "<<tl.x<<endl;
                        cout<<"tl y: "<<tl.y<<endl;
                        cout<<"br x: "<<br.x<<endl;
                        cout<<"br y: "<<br.y<<endl;
                        if(counts>100)
                            counts=counts%10+2;
                        static int tl_x,tl_y,br_x,br_y;
                        if(counts==1)
                        {
                            tl_x=tl.x;
                            tl_y=tl.y;
                            br_x=br.x;
                            br_y=br.y;
                            cout<<"static int tl x: "<<tl_x<<endl;
                            cout<<"static int tl y: "<<tl_y<<endl;
                            cout<<"static int br x: "<<br_x<<endl;
                            cout<<"static int br y: "<<br_y<<endl;
                        }
                        else
                        {  
                            if(abs(tl_x-tl.x)<10&&abs(tl_y-tl.y)<10&&abs(br_x-br.x)<10&&abs(br_y-br.y)<10)
                                continue;
                        }
                        Rect roi(tl,br);
                        Mat ROI_image;
                        src(roi).copyTo(ROI_image);
                        int black=Color_judge(ROI_image,(br.x-tl.x)*(br.y-tl.y));
                        if(!black)  continue;
                        imshow("ROI",ROI_image);
//                         Mat mask = Mat::zeros(src.size(), CV_8UC1);
//                         const Point* p = &approx[0];
//                         int n = 4;
//                         polylines(mask, &p, &n, 1, true, Scalar(255,255,255  ), 1, CV_AA);
//                         imshow("mask",mask);
//                         Point seed;
//                         seed.x=(tl.x+br.x)/2;
//                         seed.y=(tl.y+br.y)/2;
//                         floodFill(mask, seed, 255, NULL, cvScalarAll(0), cvScalarAll(0), CV_FLOODFILL_FIXED_RANGE);
                        //mask(rect).setTo(255);
//                         Mat maskImage;
//                         src.copyTo(maskImage, mask);
//                         imshow("mask_final",maskImage);


                        if('q'==(char)waitKey(5))
                            exit(0);
                        squares.push_back(approx);
                        //cout<<"approx: "<<approx<<endl;
                    }
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 1, CV_AA);
		for (int j=0;j<n;j++)
		{
			circle(image,p[j],10,Scalar(255,0,0),5);
		}
    }
    cout<<"squares.size: "<<squares.size()<<endl;
    //cout<<"squares[0]: "<<squares[0][0]<<endl<<squares[0][1]<<endl<<squares[0][2]<<endl<<squares[0][3]<<endl;
    //cout<<"square[0]: "<<rect0<<endl;
    if(squares.size()>0)
        {
            Point tl,tr,br,bl;
            vector<Point> rect=squares[0];
            MyPoint p1(rect[0]);
            MyPoint p2(rect[1]);
            MyPoint p3(rect[2]);
            MyPoint p4(rect[3]);
            vector<MyPoint> my_rect;
            my_rect.push_back(p1);
            my_rect.push_back(p2);
            my_rect.push_back(p3);
            my_rect.push_back(p4);
            sort(my_rect.begin(),my_rect.end());
            if(my_rect[0].y<my_rect[1].y)
            {
                tl.x=my_rect[0].x;
                tl.y=my_rect[0].y;
                bl.x=my_rect[1].x;
                bl.y=my_rect[1].y;
            }
            else
            {
                tl.x=my_rect[1].x;
                tl.y=my_rect[1].y;
                bl.x=my_rect[0].x;
                bl.y=my_rect[0].y;
            }
            if(my_rect[2].y<my_rect[3].y)
            {
                tr.x=my_rect[2].x;
                tr.y=my_rect[2].y;
                br.x=my_rect[3].x;
                br.y=my_rect[3].y;
            }
            else
            {
                tr.x=my_rect[3].x;
                tr.y=my_rect[3].y;
                br.x=my_rect[2].x;
                br.y=my_rect[2].y;
            }
            Calcu_attitude(world_pnt_tl,world_pnt_tr,world_pnt_br,world_pnt_bl,tl,tr,br,bl);
            cout<<"tl: "<<tl<<endl;
            cout<<"tr: "<<tr<<endl;
            cout<<"br: "<<br<<endl;
            cout<<"bl: "<<bl<<endl;
//             cout<<"square[0][0]: "<<squares[0][0]<<endl;
//             cout<<"square[0][1]: "<<squares[0][1]<<endl;
//             cout<<"square[0][2]: "<<squares[0][2]<<endl;
//             cout<<"square[0][3]: "<<squares[0][3]<<endl;
            char str_y[20];
            char str_z[20];
            char str_x[20];
            char str_tz[20];
            char str_ty[20];
            char str_tx[20];
            sprintf(str_y,"%lf",ry);
            sprintf(str_z,"%lf",rz);
            sprintf(str_x,"%lf",rx);
            sprintf(str_tz,"%lf",tz);
            sprintf(str_ty,"%lf",ty);
            sprintf(str_tx,"%lf",tx);
            string pre_str_y="thetay: ";
            string pre_str_z="thetaz: ";
            string pre_str_x="thetax: ";
            string pre_str_tz="tz: ";
            string pre_str_ty="ty: ";
            string pre_str_tx="tx: ";
            string full_y=pre_str_y+str_y;
            string full_z=pre_str_z+str_z;
            string full_x=pre_str_x+str_x;
            string full_tz=pre_str_tz+str_tz;
            string full_ty=pre_str_ty+str_ty;
            string full_tx=pre_str_tx+str_tx;
            putText(image,full_y,Point(30,30),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2);
            putText(image,full_z,Point(30,70),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,0),2);
            putText(image,full_x,Point(30,100),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,0),2);
            putText(image,full_tz,Point(30,130),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2);
            putText(image,full_ty,Point(30,170),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,0),2);
            putText(image,full_tx,Point(30,200),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,0),2);
        }
    imshow(wndname, image);
}

int main(int /*argc*/, char** /*argv*/)
{
	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << " failed to open!" << endl;
		return 1;
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT);
    set_camera_exposure(BUFFER_EXPOSURE_VALUE);
    Mat frame;
    capture.read(frame);
    capture.read(frame);
    capture.read(frame);
	
    namedWindow( wndname, 1 );
    vector<vector<Point> > squares;

   while(capture.read(frame))
    {
        double t = (double)getTickCount();
		capture >> frame;
		if (frame.empty())
			break;
        Mat src=frame.clone();
        findSquares(src,frame, squares);
        drawSquares(frame, squares);
		//imshow(wndname, frame);
        int c = waitKey(1);
        t = ((double)getTickCount() - t)/getTickFrequency();
        cout<<"time: "<<t<<" second"<<endl;
        if( (char)c == 'q' )
            break;
    }
    return 0;
}

void Calcu_attitude(Point3f world_pnt_tl,Point3f world_pnt_tr,Point3f world_pnt_br,Point3f world_pnt_bl,Point2f pnt_tl_src,Point2f pnt_tr_src,Point2f pnt_br_src,Point2f pnt_bl_src)
{
    vector<Point3f> Points3D;
    vector<Point2f> Points2D;
    // Vec3d  rvec,tvec;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    Points3D.push_back(world_pnt_tl);
    Points3D.push_back(world_pnt_tr);
    Points3D.push_back(world_pnt_br);
    Points3D.push_back(world_pnt_bl);
    Points2D.push_back(pnt_tl_src);
    Points2D.push_back(pnt_tr_src);
    Points2D.push_back(pnt_br_src);
    Points2D.push_back(pnt_bl_src);
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
    ry=thetay;
    rz=thetaz;
    rx=thetax;
    tx=tvec.ptr<double>(0)[0];
    ty=tvec.ptr<double>(1)[0];
    tz=tvec.ptr<double>(2)[0];
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
int Color_judge(Mat src,int area)
{
    Mat grey;
    cvtColor(src,grey,CV_BGR2GRAY);
    const int channels[1]= {0};
    const int histSize[1]= {256};
    float hranges[2]= {0,255};
    const float* ranges[1]= {hranges};
    MatND hist;
    calcHist(&grey,1,channels,Mat(),hist,1,histSize,ranges);
    //cout<<"hist: "<<hist<<endl;
    float sum=0;
    for(int i=0; i<grey_thresh; i++)
    {
        sum+=hist.at<float>(i);
    }
    cout<<"sum: "<<sum<<endl;
    cout<<"portion: "<<sum/area<<endl;
    return sum/area>0.7?1:0;
}
