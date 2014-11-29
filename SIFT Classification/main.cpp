#include<cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<iostream>

#include<Sift.h>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    //freopen("out.txt", "w", stdout);
    Mat img = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
    if(img.empty())
       return -1;
    Mat gimg;
    Mat norm = Mat::zeros(img.rows, img.cols, CV_32F);
    cvtColor(img, gimg, CV_RGB2GRAY);
    int max1 = -1;
    for (int i=0;i<gimg.rows;i++) {
        for (int j=0;j<gimg.cols;j++) {
            max1 = max(max1, (int)gimg.at<uchar>(i, j));
        }
    }
    for (int i=0;i<gimg.rows;i++) {
        for (int j=0;j<gimg.cols;j++) {
            norm.at<float>(i, j)= gimg.at<uchar>(i, j)*1.0/max1;
        }
    }
    //cout<<norm<<endl;
    Sift sift;
    vector<vector<Mat> > pyr, dog;
    vector<KeyPoint> kp;
    cout<<"Start"<<endl;
    sift.buildGaussianPyramid(norm, pyr, 4);
    cout<<"Hop1"<<endl;
    dog = sift.buildDogPyr(pyr);
    cout<<"Hop2"<<endl;
    sift.getScaleSpaceExtrema(dog, kp);
    cout<<"Size Kp: "<<kp.size()<<endl;
    sift.cleanPoints(dog, kp, 10);
    /*int sum = 0;
    for (int i=0;i<dog.size();i++)
        for (int j=0;j<dog[i].size();j++)
            for (int r = 0;r<dog[i][j].rows;r++)
                for (int c = 0;c<dog[i][j].cols;c++)
                    if (dog[i][j].at<float>(r,c) >= 0.03)
                        sum++;
    cout<<"Sum: "<<sum<<endl;*/
    int sum = 0;
    for (int i=0;i<kp.size();i++) {
        if (kp[i].octave==0) {
      //      cout<<"Found"<<endl;
            sum++;
            gimg.at<uchar>(kp[i].pt.x, kp[i].pt.y) = 255;
        }
    }
    cout<<"Sum: "<<sum<<endl;
    namedWindow("lena", CV_WINDOW_AUTOSIZE );
    imshow("lena", gimg);
    waitKey(0);
    return 0;
}
