#include "Sift.h"

#include<cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<iostream>
#include <math.h>

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

using namespace cv;
using namespace std;

// TODO define the number of blur levels
int blur_levels = 5;
double initial_sigma = 1.6;
double blurring_factor = 1.41421;

Size gaussian_kernel_size(3,3);

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

void Sift::findSiftInterestPoint(Mat& image, vector<KeyPoint>&  )
{
}

void Sift::buildGaussianPyramid(Mat& image, vector< vector <Mat> >& pyr, int nOctaves)
{
    pyr.push_back(vector<Mat>());
    pyr[0].push_back(image);
    for (int i = 0; i < nOctaves; ++i) {
        double sigma = initial_sigma;
        for (int j = 1; j < blur_levels; ++j) {
            Mat new_img;
            GaussianBlur(pyr[i][j-1], new_img, gaussian_kernel_size, sigma, 0);
            pyr[i].push_back(new_img);
            sigma *= blurring_factor;
        }
        pyr.push_back(vector<Mat>());
        pyr[i+1].push_back(downSample(pyr[i][1]));
    }
}

void Sift::cleanPoints(Mat& image, int curv_thr)
{
}

//based on contrast //and principal curvature ratio
Mat Sift::downSample(Mat& image)
{
    Size new_mat_size = image.size();
    new_mat_size.width>>=1;
    new_mat_size.height>>=1;
    Mat new_image = Mat::zeros(new_mat_size, image.type());
    for (int i = 1; i < image.rows; i += 2) {
        int k = 0;
        for (int j = 1; j < image.cols; j += 2) {
            new_image.at<uchar>(i>>1, k++) = image.at<uchar>(i, j);
        }
    }
    return new_image;
}

vector<vector<Mat> > Sift::buildDogPyr(vector<vector<Mat> > gauss_pyr)
{
    vector<vector<Mat> > dog_pyr;
    for (int i = 0; i < gauss_pyr.size(); ++i) {
        dog_pyr.push_back(vector<Mat>());
        for (int j = 0; j < gauss_pyr[i].size() - 1; ++j) {
            dog_pyr[i].push_back(gauss_pyr[i][j] - gauss_pyr[i][j+1]);
        }
    }
    // <test_code>
    vector<KeyPoint> k;
    getScaleSpaceExtrema(dog_pyr, k);
    // <\test_code>
    return dog_pyr;
}

vector<double> Sift::computeOrientationHist(const Mat& image)
{
}

// Calculates the gradient vector of the feature
void Sift::getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints)
{
    for (int i = 0; i < dog_pyr.size(); i++) {
        for (int j = 1; j < dog_pyr[i].size() - 1; j++) {
            Mat myMat = Mat::zeros(dog_pyr[i][j].size(), dog_pyr[i][j].type());
            for (int r = 1; r < dog_pyr[i][j].rows - 1; r++) {
                for (int c = 1; c < dog_pyr[i][j].cols - 1; c++) {
                    if (isLocalExtrema(dog_pyr[i][j - 1], dog_pyr[i][j], dog_pyr[i][j + 1], r, c)) {
                        float dx = (float) (dog_pyr[i][j].at<uchar>(r + 1,c) - dog_pyr[i][j].at<uchar>(r - 1,c));
                        float dy = (float) (dog_pyr[i][j].at<uchar>(r,c + 1) - dog_pyr[i][j].at<uchar>(r,c - 1));
                        float orientation = (float) atan(dy/dx);
                        float magnitude = (float) sqrt((dx*dx) + (dy*dy));
                        KeyPoint key_point(r, c, 3, orientation, magnitude, i);
                        keypoints.push_back(key_point);
                    }
                }
            }
            imshow("lena", myMat);
        }
    }
}

bool Sift::isLocalExtrema(Mat& img_above, Mat& img, Mat& img_below, int x, int y) {
    int dx[9] = {-1,0,1,-1,0,1,-1, 0, 1};
    int dy[9] = { 1,1,1, 0,0,0,-1,-1,-1};
    uchar val = img.at<uchar>(x,y);
    for (int i = 0; i < ARRAY_SIZE(dx); i++) {
        if (img.at<uchar>(x + dx[i], y + dy[y]) > val ||
            img_above.at<uchar>(x + dx[i], y + dy[y]) > val ||
            img_below.at<uchar>(x + dx[i], y + dy[y]) > val) {
                return false;
            }
    }
    return true;
}
