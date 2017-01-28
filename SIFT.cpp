#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>

using namespace cv;


int main() {
    Mat image, grad;
    image = imread("298.jpg");
    
    std::vector<KeyPoint> keypoints;
    
    SurfFeatureDetector detector(3200);
    detector.detect(image, keypoints);
    
    //draw points on picture
    Mat img_keypoints;
    drawKeypoints(image, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    namedWindow("rtr", WINDOW_AUTOSIZE);
    imshow("rtr", img_keypoints);
    imwrite("new_SIFT.jpg", img_keypoints);
    waitKey(0);
    return 0;
}
