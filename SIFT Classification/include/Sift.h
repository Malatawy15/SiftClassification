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
        void cleanPoints( Mat& image, int curv_thr );
        //based on contrast //and principal curvature ratio
        Mat downSample(Mat& image);
        vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr);
        vector<double> computeOrientationHist(const Mat& image);
        // Calculates the gradient vector of the feature
        void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);
    protected:
    private:
};

#endif // SIFT_H
