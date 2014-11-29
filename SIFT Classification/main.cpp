#include<cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<iostream>

#include<Sift.h>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Mat img = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
    if(img.empty())
       return -1;
    Mat gimg;
    Mat norm = Mat::zeros(img.rows, img.cols, CV_32F);
    cvtColor(img, gimg, CV_RGB2GRAY);
    int max1 = -1;
    for (int i=0;i<gimg.rows;i++) {
        for (int j=0;j<gimg.cols;j++) {
            max1 = max(max1, (int)gimg.at<uchar>(i, j));
        }
    }
    for (int i=0;i<gimg.rows;i++) {
        for (int j=0;j<gimg.cols;j++) {
            norm.at<float>(i, j)= gimg.at<uchar>(i, j)*1.0/max1;
        }
    }
    Sift sift;
    vector<KeyPoint> kp;
    sift.findSiftInterestPoint(norm, kp);
    /*vector<double> vd = sift.computeOrientationHist(norm, kp);
    for (int i=0;i<vd.size();i++)
        cout<<vd[i]<<endl;*/
    int sum = 0;
    for (int i=0;i<kp.size();i++) {
        if (kp[i].octave==0) {
            sum++;
            gimg.at<uchar>(kp[i].pt.x, kp[i].pt.y) = 255;
        }
    }
    cout<<"Sum: "<<sum<<endl;
    namedWindow("lena", CV_WINDOW_AUTOSIZE );
    imshow("lena", gimg);
    waitKey(0);
    return 0;
}
