#ifndef SIFT_H
#define SIFT_H

#include<cv.h>

using namespace cv;

class Sift
{
    public:
        Sift();
        virtual ~Sift();

        void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints);

        void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, int nOctaves);
        Mat downSample(Mat& image);
        vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr);

        void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);

        void cleanPoints(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints, int curv_thr);

        vector<double> computeOrientationHist(const Mat& image);

    protected:
    private:
        bool isLocalExtrema(Mat& img_above, Mat& img, Mat& img_below, int x, int y);
};

#endif // SIFT_H
