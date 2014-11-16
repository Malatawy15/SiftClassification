#include "Sift.h"

#include<cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<iostream>

using namespace cv;
using namespace std;

Sift::Sift()
{
    //ctor
}

Sift::Sift(string image_path, bool is_color)
{
    if (is_color) {
        input_image = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
    } else {
        input_image = imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    }
}

Sift::~Sift()
{
    //dtor
}

void Sift::findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints)
{
}

void Sift::buildGaussianPyramid(Mat& image, vector< vector <Mat> >& pyr, int nOctaves)
{
}

void Sift::cleanPoints(Mat& image, int curv_thr)
{
}

//based on contrast //and principal curvature ratio
Mat Sift::downSample(Mat& image)
{
    Size new_mat_size = image.size();
    std::cout<<(image.size())<<std::endl;
    new_mat_size.width>>=1;
    new_mat_size.height>>=1;
    Mat new_image = Mat::zeros(new_mat_size, image.type());
    std::cout<<new_mat_size<<std::endl;
//    cout<<new_image<<endl;
    for (int i = 1; i < image.rows; i += 2) {
        int k = 0;
        for (int j = 1; j < image.cols; j += 2) {
            new_image.at<uchar>(i>>1, k++) = image.at<uchar>(i, j);
        }
    }
    cout<<"Done"<<endl;
 //   cout<<new_image<<endl;
    return new_image;
    //Mat new_img;
    //pyrDown(image, new_img, new_mat_size);
    //return new_img;
}

vector<vector<Mat> > Sift::buildDogPyr(vector<vector<Mat> > gauss_pyr)
{
}

vector<double> Sift::computeOrientationHist(const Mat& image)
{
}

// Calculates the gradient vector of the feature
void Sift::getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints)
{
}
