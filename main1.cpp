#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


int main() {
    Mat image;
    image = imread("1095.jpg");
    
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                image.at<Vec3b>(y,x)[c] = 255 - image.at<Vec3b>(y,x)[c];
            }
        }
    }
    
    Point center = Point(image.cols/2, image.rows/2);
    double angle = 30.0;
    double scale = 2/3.0;
    
    Mat rot_mat(2, 3, CV_32FC1);
    rot_mat = getRotationMatrix2D(center, angle, scale);
    
    Mat change;
    
    warpAffine(image, change, rot_mat, image.size() );
    
    
    namedWindow("rtr", WINDOW_AUTOSIZE);
    imshow("rtr", change);
    imwrite("new_change.jpg", change);
    waitKey(0);
    return 0;
}
