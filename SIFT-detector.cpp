#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>

using namespace cv;


int main() {
    Mat image1, image2, image3;
    
    image2 = imread("1.jpg");
    
    image3 = imread("new.jpg");
    
    //create rotate
    Point center = Point(image2.cols/2, image2.rows/2);
    double angle = 30.0;
    double scale = 2/3.0;
    Mat rot_mat(2, 3, CV_32FC1);
    rot_mat = getRotationMatrix2D(center, angle, scale);
    warpAffine(image2, image1, rot_mat, image2.size() );
    
    //detect keypoints
    SiftFeatureDetector detector;
    
    std::vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(image3, keypoints1);
    detector.detect(image2, keypoints2);
    
    //calculate descriptors
    SiftDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    
    extractor.compute(image3, keypoints1, descriptors1);
    extractor.compute(image2, keypoints2, descriptors2);
    
    //matching descriptor
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    //calculate distance
    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < descriptors1.rows; ++i)
    { double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    
    std::vector<DMatch> good_matches;
    for( int i = 0; i < descriptors1.rows; i++ )
        { if( matches[i].distance <= max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
    }
    
    //draw good matches
    Mat img_matches;
    drawMatches(image3, keypoints1, image2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("rtr", WINDOW_AUTOSIZE);
    imshow("rtr", img_matches);
    imwrite("new_matches_rotate.jpg", img_matches);
    waitKey(0);
    return 0;
}