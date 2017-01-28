#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;

int main() {
    Mat image;
    image = imread("6379.jpg");
    

    Mat image1, image2;
    
    image2 = imread("6379.jpg");
    
    //reduce noise
    GaussianBlur(image2, image2, Size(5,5), 5, 0, BORDER_DEFAULT );
    
    //create rotate
    Point center = Point(image2.cols/2, image2.rows/2);
    double angle = 30.0;
    double scale = 1;
    Mat rot_mat(2, 3, CV_32FC1);
    rot_mat = getRotationMatrix2D(center, angle, scale);
    warpAffine(image2, image1, rot_mat, image2.size() );

    namedWindow("rtr", WINDOW_AUTOSIZE);
    imshow("rtr", image1);
    imwrite("new.jpg", image1);
    waitKey(0);
    return 0;
}
