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

Size gaussian_kernel_size(3,3);

Sift::Sift()
{
    //ctor
}

Sift::~Sift()
{
    //dtor
}
/*
@params:
    image: input image
    keypoints: output keypoints
@description:
    Function that linearly calls all othe helper functions to
    run the sift algorithm.
@return
    -
*/
void Sift::findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints)
{
    vector<vector<Mat> > pyr;
    buildGaussianPyramid(image, pyr, Sift::number_of_octaves);
    vector<vector<Mat> > dog_pyramid = buildDogPyr(pyr);
    getScaleSpaceExtrema(dog_pyramid, keypoints);
    cleanPoints(dog_pyramid, keypoints, Sift::principal_curvature_threshold);
}

/*
@params:
    image: input image
    pyr: output gaussian pyramid
    nOctaves: number of octaves produced
@description:
    Creates the gaussian pyramid from the input image. Downsamples the input image
    the number of times specified by the global variable "blur-levels". Moreover
    the number of octaves created are equal to the input nOctaves
@return:
    -
*/
void Sift::buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, int nOctaves)
{
    pyr.push_back(vector<Mat>());
    pyr[0].push_back(image);
    for (int i = 0; i < nOctaves; ++i) {
        double sigma = Sift::initial_sigma;
        for (int j = 1; j < Sift::blur_levels; ++j) {
            Mat new_img = Mat::zeros(pyr[i][j-1].size(), CV_32F);
            GaussianBlur(pyr[i][j-1], new_img, gaussian_kernel_size, sigma, 0);
            pyr[i].push_back(new_img);
            sigma *= Sift::blurring_factor;
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

/*
@params:
    image: input image to be down sampled
@description:
    Downsamples the the input image to half it's
    height and width
@return:
    Mat: output matrix containing the downsampled image
*/
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

/*
@params:
    gauss_pyr: A vector (representing octaves) including a vector (containing
               the blur levels) of images
@description:
    Subtracts each image in the gaussian pyramid from the image bellow it in
    the vector of blur levels.
@return:
    vector<vector<Mats> >: contains each difference of gaussian
*/
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
    //vector<KeyPoint> k;
    //getScaleSpaceExtrema(dog_pyr, k);
    // <\test_code>
    //imshow("img_above", dog_pyr[1][0]);
    //imshow("img", dog_pyr[1][1]);
    //imshow("img_below", dog_pyr[1][2]);
    return dog_pyr;
}

/*
@params:
    dog_pyr: input difference of gaussian pyramid
    keypoints: output maxima and minima keypoints
@description:
    Returns a vector of keypoints containing any point that is an extrema
    in it's surrounding 27 pixels
@return:
    -
*/
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

/*
@params:
    img_above: The image above our candidate image blur level
    img: The candidate image
    img_below: The image below the candidate image
    x: x-coordinate of candidate pixel
    y: y-coordinate of candidate pixel
@description:
    Helper method for getScaleSpaceExtrema. Determines whether the
    candidate pixel is an extreme in the 8 pixels in it's own image
    and the 9 pixels in both the image above and elow it.
@return:
    bool: returns whether the pixel is an extrema or not.
*/
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

/*
@params:
    dog_pyr: The difference of gaussian pyramid
    keypoints: The candidate keypoints. Filtering happens in place
    curv_thr: The principal curvature threshold "r"
@description:
    This method cleans the vector of candidate keypoints by filtering out the points
    that are less than the contrast threshold then filters out the poorly localized
    keypoints by calculating the equation TH(r)^2 / Det(r) then comparing it to the
    principal curvature threshold using the equation (r+1)^2 / r.
@return:
    void - filtering keypoints happens in place in the vector keypoints
*/
void Sift::cleanPoints(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints, int curv_thr)
{
    cout<<"Size Initial: "<<keypoints.size()<<endl;
    vector<KeyPoint>::iterator it = keypoints.begin();
    double principal_curvature_threshold_value = ((curv_thr+1.0)*(curv_thr+1.0)/curv_thr);
    while (it != keypoints.end()){
        float intensity_value = abs(dog_pyr[it->octave][it->class_id].at<float>(it->pt.x, it->pt.y));
        if (intensity_value < Sift::intensity_threshold){
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

/*
@params:
    image: input image
    keypoints: vector of filtered keypoints for the image
@description:
    Returns orientation histogram for the image
@return:
    vector<double>: contains the descriptors for all the keypoints.
*/
vector<double> Sift::computeOrientationHist(const Mat& image, vector<KeyPoint>& keypoints)
{
    vector<double> result;
    for (int i = 0;i < keypoints.size(); ++i) {
        vector<double>res;
        if(computeOrientationHistAtPoint(image, keypoints[i].pt.x, keypoints[i].pt.y, res)) {
            result.insert(result.end(), res.begin(), res.end());
        }
    }
    return result;
}

/*
@params:
    image: input image
    x: x-coordinate of the keypoint
    y: y-coordinate of the keypoint
    descriptors: 128 descriptor of the keypoint
@description:
    Calculates the descriptors of a single keypoint
@return:
    bool: whether or not the histogram is computable for this pixel
*/bool Sift::computeOrientationHistAtPoint(const Mat& image, int x, int y, vector<double>& descriptor)
{
    int maxX = image.size().width;
    int maxY = image.size().height;
    if (x < 8 || x > maxX - 8 || y < 8 || y > maxY - 8) {
        return false;
    } else {
        int block_count_x = 0;
        for (int i = x - 7; block_count_x < 4; i += 4) {
            int block_count_y = 0;
            for (int j = x + 7; block_count_y < 4; y += 4) {
                vector<double> partial_result = computePartialHistogram(image, i, j);
                for (int k = 0; k < partial_result.size(); k++) {
                    descriptor.push_back(partial_result.at(k));
                }
                block_count_y++;
            }
            block_count_x++;
        }
        return true;
    }
}

/*
@params:
    image: input image
    x: x-coordinate of keypoint quartile
    y: y-coordinate of keypoint quartile
@description:
    computes an 8-bin histogram of descriptors on the
    quartile of the 16x16 pixels surrounding the keypoint
@return:
    vector<double>: 8-bin histogram of descriptors
*/
vector<double> Sift::computePartialHistogram(const Mat& image, int x, int y)
{
    vector<double> result;
    for (int i = 0; i < 8; i++) {
        result.push_back(0.0);
    }
    for (int i = x; i < x + 4; i++) {
        for (int j = y; j < y + 4; j++) {
            double dx = ((double) image.at<uchar>(i + 1,j) - (double) image.at<uchar>(i - 1,j));
            double dy = ((double) image.at<uchar>(i,j + 1) - (double) image.at<uchar>(i,j - 1));
            double orientation = (double) atan(dy/dx);
            double magnitude = (double) sqrt((dx*dx) + (dy*dy));
            result.at((int) orientation/45) += magnitude;
        }
    }
    return result;
}
