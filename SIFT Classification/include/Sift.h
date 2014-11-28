#ifndef SIFT_H
#define SIFT_H

#include<cv.h>

using namespace cv;

class Sift
{
    public:
        Mat input_image;
        Sift();
        Sift(string image_path, bool is_color);
        virtual ~Sift();
        void findSiftInterestPoint( Mat& image, vector<KeyPoint>& keypoints);

        void buildGaussianPyramid( Mat& image, vector< vector <Mat> >& pyr, int nOctaves );
        void cleanPoints(Mat& image, vector<KeyPoint>& keypoints, int curv_thr );
        //based on contrast //and principal curvature ratio
        Mat downSample(Mat& image);
        vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr);
        bool computeOrientationHist(const Mat& image, int x, int y, vector<double> descriptor);
        vector<double> computePartialHistogram(const Mat& image, int x, int y);
        // Calculates the gradient vector of the feature
        void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);
    protected:
    private:
        bool isLocalExtrema(Mat& img_above, Mat& img, Mat& img_below, int x, int y);
};

#endif // SIFT_H
