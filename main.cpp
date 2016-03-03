#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;

int main() {
    Mat image;
    image = imread("1.jpg");
    

    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                image.at<Vec3b>(y,x)[c] = 255 - image.at<Vec3b>(y,x)[c];
            }
        }
    }

    namedWindow("rtr", WINDOW_AUTOSIZE);
    imshow("rtr", image);
    imwrite("new.jpg", image);
    waitKey(0);
    return 0;
}
