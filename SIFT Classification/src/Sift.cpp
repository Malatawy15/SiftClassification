#include "Sift.h"

#include<cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

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
