#ifndef SIFT_H
#define SIFT_H

#include<cv.h>

using namespace cv;

class Sift
{
    public:
        static const int blur_levels = 5;
        static const double initial_sigma = 1.41421;
        static const double blurring_factor = 2;
        static const double intensity_threshold = 0.003;
        static const int principal_curvature_threshold = 10;
        static const int number_of_octaves = 4;

        Sift();
        virtual ~Sift();

        void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints);

        void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, int nOctaves);
        Mat downSample(Mat& image);
        vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr);

        void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);

        void cleanPoints(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints, int curv_thr);

        vector<double> computeOrientationHist(const Mat& image, vector<KeyPoint>& keypoints);

    protected:
    private:
        bool isLocalExtrema(Mat& img_above, Mat& img, Mat& img_below, int x, int y);
        vector<double> computePartialHistogram(const Mat& image, int x, int y);
        bool computeOrientationHistAtPoint(const Mat& image, int x, int y, vector<double>& descriptor);
};

#endif // SIFT_H
