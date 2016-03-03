#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


int main() {
    Mat image, grad;
    image = imread("1.jpg");
    
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    //reduce noise
    GaussianBlur(image, image, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    //x-gradient with Sobel opertor
    Sobel(image, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs(grad_x, abs_grad_x);
    
    //y-gradient with Sobel operator
    Sobel(image, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs(grad_y, abs_grad_y);
    
    //average xy-gradient with Sobel operator
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    
    
    
    namedWindow("rtr", WINDOW_AUTOSIZE);
    imshow("rtr", abs_grad_x);
    imwrite("new_change.jpg", abs_grad_x);
    waitKey(0);
    return 0;
}
