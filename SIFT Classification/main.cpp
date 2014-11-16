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
    vector<vector<Mat> > pyr;
    sift.buildGaussianPyramid(gimg, pyr, 4);
    sift.buildDogPyr(pyr);
    waitKey(0);
    return 0;
}
