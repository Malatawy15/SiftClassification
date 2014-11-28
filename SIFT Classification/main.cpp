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
    cvtColor(img, gimg, CV_RGB2GRAY);
    namedWindow("lena", CV_WINDOW_AUTOSIZE );
    imshow("lena", gimg);
    Sift sift;
    vector<KeyPoint> kp;
    sift.findSiftInterestPoint(gimg, kp);
    /*vector<vector<Mat> > pyr, dog;
    vector<KeyPoint> kp;
    sift.buildGaussianPyramid(gimg, pyr, 4);
    dog = sift.buildDogPyr(pyr);
    sift.getScaleSpaceExtrema(dog, kp);
    sift.cleanPoints(gimg, kp, 10);*/
    waitKey(0);
    return 0;
}
