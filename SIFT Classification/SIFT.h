#ifndef SIFT_H_INCLUDED
#define SIFT_H_INCLUDED

namespace SIFT {
    class Sift{
      private:
        void findSiftInterestPoint( Mat& image, vector<KeyPoint>& keypoints);
        void buildGaussianPyramid( Mat& image, vector< vector <Mat> >& pyr, int nOctaves );
        void cleanPoints( Mat& image, int curv_thr );
        //based on contrast //and principal curvature ratio
        Mat downSample(Mat& image);
        vector<vector<Mat>> buildDogPyr(vector<vector<Mat>> gauss_pyr);
        vector<double> computeOrientationHist(const Mat& image);
        // Calculates the gradient vector of the feature
        void getScaleSpaceExtrema(vector<vector<Mat>>& dog_pyr, vector<KeyPoint>& keypoints);
    };
}

#endif // SIFT_H_INCLUDED
