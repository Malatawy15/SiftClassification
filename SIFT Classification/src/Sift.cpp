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

// TODO define the number of blur levels
int blur_levels = 5;
double initial_sigma = 1.41421;
double blurring_factor = 2;
double intensity_threshold = 0.003;
int principal_curvature_threshold = 10;
int number_of_octaves = 4;

Size gaussian_kernel_size(3,3);

Sift::Sift()
{
    //ctor
}

Sift::~Sift()
{
    //dtor
}

void Sift::findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints)
{
    vector<vector<Mat> > pyr;
    buildGaussianPyramid(image, pyr, number_of_octaves);
    vector<vector<Mat> > dog_pyramid = buildDogPyr(pyr);
    getScaleSpaceExtrema(dog_pyramid, keypoints);
    cleanPoints(dog_pyramid, keypoints, principal_curvature_threshold);
}

void Sift::buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, int nOctaves)
{
    pyr.push_back(vector<Mat>());
    pyr[0].push_back(image);
    for (int i = 0; i < nOctaves; ++i) {
        double sigma = initial_sigma;
        for (int j = 1; j < blur_levels; ++j) {
            Mat new_img = Mat::zeros(pyr[i][j-1].size(), CV_32F);
            GaussianBlur(pyr[i][j-1], new_img, gaussian_kernel_size, sigma, 0);
            pyr[i].push_back(new_img);
            sigma *= blurring_factor;
            //namedWindow("lena", CV_WINDOW_AUTOSIZE );
            //imshow("lena", new_img);
            //waitKey(0);
        }
        if (i+1 < nOctaves) {
            pyr.push_back(vector<Mat>());
            pyr[i+1].push_back(downSample(pyr[i][1]));
        }
    }
}

Mat Sift::downSample(Mat& image)
{
    Mat new_image = Mat::zeros(image.rows/2, image.cols/2, CV_32F);
    for (int i = 1; i < image.rows; i += 2) {
        int k = 0;
        for (int j = 1; j < image.cols; j += 2) {
            new_image.at<float>(i>>1, k++) = image.at<float>(i, j);
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
    /*
    // <test_code>
    vector<KeyPoint> k;
    getScaleSpaceExtrema(dog_pyr, k);
    // <\test_code>
    imshow("img_above", dog_pyr[1][0]);
    imshow("img", dog_pyr[1][1]);
    imshow("img_below", dog_pyr[1][2]);*/
    return dog_pyr;
}

void Sift::getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints)
{
    for (int i = 0; i < dog_pyr.size(); i++) {
        for (int j = 1; j < dog_pyr[i].size() - 1; j++) {
            for (int r = 1; r < dog_pyr[i][j].rows - 1; r++) {
                for (int c = 1; c < dog_pyr[i][j].cols - 1; c++) {
                    if (isLocalExtrema(dog_pyr[i][j - 1], dog_pyr[i][j], dog_pyr[i][j + 1], r, c)) {
                        float dx = (dog_pyr[i][j].at<float>(r + 1,c) - dog_pyr[i][j].at<float>(r - 1,c));
                        float dy = (dog_pyr[i][j].at<float>(r,c + 1) - dog_pyr[i][j].at<float>(r,c - 1));
                        //cout<<"dx: "<<dx<<" dy:"<<dy<<endl;
                        float orientation = (float) atan(dy/dx);
                        float magnitude = (float) sqrt((dx*dx) + (dy*dy));
                        KeyPoint key_point(r, c, 3, orientation, magnitude, i, j);
                        keypoints.push_back(key_point);
                    }
                }
            }
        }
    }
    Mat myMat = Mat::zeros(dog_pyr[1][0].size(), dog_pyr[1][0].type());
    for (int i = 0; i < keypoints.size(); i++) {
        if (keypoints[i].octave == 2) {
            myMat.at<uchar>(keypoints[i].pt.x, keypoints[i].pt.y) = keypoints[i].response;
            cout<< keypoints[i].response<<endl;
        }
    }
    //imshow("bla", myMat);

}

bool Sift::isLocalExtrema(Mat& img_above, Mat& img, Mat& img_below, int x, int y) {
    int dx[9] = {-1,0,1,-1,0,1,-1, 0, 1};
    int dy[9] = { 1,1,1, 0,0,0,-1,-1,-1};
    int dx_same_level[9] = {-1,0,1,-1,-1,1,-1, 0, 1};
    int dy_same_level[9] = { 1,1,1, 0, 1,0,-1,-1,-1};
    float val = img.at<float>(x,y);
    for (int i = 0; i < ARRAY_SIZE(dx); i++) {
        if ((img.at<float>(x + dx_same_level[i], y + dy_same_level[i]) >= val ||
            img_above.at<float>(x + dx[i], y + dy[i]) >= val ||
            img_below.at<float>(x + dx[i], y + dy[i]) >= val) &&
            (img.at<float>(x + dx_same_level[i], y + dy_same_level[i]) <= val ||
            img_above.at<float>(x + dx[i], y + dy[i]) <= val ||
            img_below.at<float>(x + dx[i], y + dy[i]) <= val)) {
                return false;
            }
    }
    return true;
}

void Sift::cleanPoints(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints, int curv_thr)
{
    cout<<"Size Initial: "<<keypoints.size()<<endl;
    vector<KeyPoint>::iterator it = keypoints.begin();
    double principal_curvature_threshold_value = ((curv_thr+1.0)*(curv_thr+1.0)/curv_thr);
    while (it != keypoints.end()){
        float intensity_value = abs(dog_pyr[it->octave][it->class_id].at<float>(it->pt.x, it->pt.y));
        if (intensity_value < intensity_threshold){
            it = keypoints.erase(it);
        }
        else {
            it++;
        }
    }
    cout<<"Size Intensity: "<<keypoints.size()<<endl;
    it = keypoints.begin();
    while (it != keypoints.end()){
        int x = it->pt.x;
        int y = it->pt.y;
        float dxx = dog_pyr[it->octave][it->class_id].at<float>(x-1, y) * 1 +
                    dog_pyr[it->octave][it->class_id].at<float>(x, y) * -2 +
                    dog_pyr[it->octave][it->class_id].at<float>(x+1, y) * 1;
        float dyy = dog_pyr[it->octave][it->class_id].at<float>(x, y-1) * 1 +
                    dog_pyr[it->octave][it->class_id].at<float>(x, y) * -2 +
                    dog_pyr[it->octave][it->class_id].at<float>(x, y+1) * 1;
        float dxy = (dog_pyr[it->octave][it->class_id].at<float>(x-1, y-1) * 1 +
                    dog_pyr[it->octave][it->class_id].at<float>(x+1, y-1) * -1 +
                    dog_pyr[it->octave][it->class_id].at<float>(x-1, y+1) * -1 +
                    dog_pyr[it->octave][it->class_id].at<float>(x+1, y+1) * 1) / 4.0;
        float thr = dxx + dyy;
        float det = dxx * dyy - dxy*dxy;
        float curvature = thr * thr / det;
        if (curvature  > principal_curvature_threshold_value){
            it = keypoints.erase(it);
        }
        else {
            it++;
        }
    }
    cout<<"Size Final: "<<keypoints.size()<<endl;
}

vector<double> Sift::computeOrientationHist(const Mat& image)
{
}
