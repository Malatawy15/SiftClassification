#include "Sift.h"

#include<cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<iostream>
#include <math.h>
#include<algorithm>

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

using namespace cv;
using namespace std;

int blur_levels = 5;
double initial_sigma = 1.6;
double blurring_factor = 1.41421;
double intensity_threshold = 0.03;
int principal_curvature_threshold = 10;
int num_of_octaves = 4;
// TODO: add vavlue to curv_thr
int curv_thr = 0;

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

void Sift::findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints)
{
    vector<vector<Mat> > gauss_pyr;
    buildGaussianPyramid(image, gauss_pyr, num_of_octaves);
    vector<vector<Mat> > dog_pyr = buildDogPyr(gauss_pyr);
    getScaleSpaceExtrema(dog_pyr, keypoints);
    cleanPoints(image, keypoints, curv_thr);
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
        if (i+1 < nOctaves) {
            pyr.push_back(vector<Mat>());
            pyr[i+1].push_back(downSample(pyr[i][1]));
        }
    }
}

double edge_ratio(Mat& image, KeyPoint kp)
{
    Point p = kp.pt;
    double dx = ((float) image.at<uchar>(p.x + 1, p.y) - (float) image.at<uchar>(p.x - 1, p.y));
    double dy = ((float) image.at<uchar>(p.x, p.y + 1) - (float) image.at<uchar>(p.x, p.y - 1));
    double dxx = dx * dx;
    double dyy = dy * dy;
    double dxy = dx * dy;
    double thr = dxx + dyy;
    double det = dxx * dyy - dxy*dxy;
    return thr * thr / det;
}

void Sift::cleanPoints(Mat& image, vector<KeyPoint>& keypoints, int curv_thr)
{
    cout<<"Size: "<<keypoints.size()<<endl;
    vector<KeyPoint>::iterator it = keypoints.begin();
    while (it != keypoints.end()){
        if (abs(it->response) < intensity_threshold || edge_ratio(image, *it) > (curv_thr+1.0)*(curv_thr+1.0)/curv_thr) {
            it = keypoints.erase(it);
        }
        else {
            it++;
        }
    }
    cout<<"Size: "<<keypoints.size()<<endl;
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
    imshow("img_above", dog_pyr[1][0]);
    imshow("img", dog_pyr[1][1]);
    imshow("img_below", dog_pyr[1][2]);
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
            for (int r = 1; r < dog_pyr[i][j].rows - 1; r++) {
                for (int c = 1; c < dog_pyr[i][j].cols - 1; c++) {
                    if (isLocalExtrema(dog_pyr[i][j - 1], dog_pyr[i][j], dog_pyr[i][j + 1], r, c)) {
                        float dx = ((float) dog_pyr[i][j].at<uchar>(r + 1,c) - (float) dog_pyr[i][j].at<uchar>(r - 1,c));
                        float dy = ((float) dog_pyr[i][j].at<uchar>(r,c + 1) - (float) dog_pyr[i][j].at<uchar>(r,c - 1));
                        //cout<<"dx: "<<dx<<" dy:"<<dy<<endl;
                        float orientation = (float) atan(dy/dx);
                        float magnitude = (float) sqrt((dx*dx) + (dy*dy));
                        KeyPoint key_point(r, c, 3, orientation, magnitude, i + 1);
                        keypoints.push_back(key_point);
                    }
                }
            }
        }
    }
    // Test code showing pre-pruning keypoints
    /*Mat myMat = Mat::zeros(dog_pyr[1][0].size(), dog_pyr[1][0].type());
    for (int i = 0; i < keypoints.size(); i++) {
        if (keypoints[i].octave == 2) {
            myMat.at<uchar>(keypoints[i].pt.x, keypoints[i].pt.y) = 255;
        }
    }
    imshow("bla", myMat);*/

}

bool Sift::isLocalExtrema(Mat& img_above, Mat& img, Mat& img_below, int x, int y) {
    int dx[9] = {-1,0,1,-1,0,1,-1, 0, 1};
    int dy[9] = { 1,1,1, 0,0,0,-1,-1,-1};
    int dx_same_level[9] = {-1,0,1,-1,-1,1,-1, 0, 1};
    int dy_same_level[9] = { 1,1,1, 0, 1,0,-1,-1,-1};
    int val = (int) img.at<uchar>(x,y);
    for (int i = 0; i < ARRAY_SIZE(dx); i++) {
        if ((int) img.at<uchar>(x + dx_same_level[i], y + dy_same_level[i]) >= val ||
            (int) img_above.at<uchar>(x + dx[i], y + dy[i]) >= val ||
            (int) img_below.at<uchar>(x + dx[i], y + dy[i]) >= val) {
                return false;
            }
    }
    return true;
}
