#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <string>

using namespace cv;


int main() {
    FileStorage database("base.yml", FileStorage::WRITE);
    
    for (int i = 298; i<10000; ++i) {
        Mat image;
        image = imread("../../../Downloads/kinopoisk/" + std::to_string(i) + ".jpg");
        
        if (image.data) {
            //detect keypoints
            SiftFeatureDetector detector;
    
            std::vector<KeyPoint> keypoints;
            detector.detect(image, keypoints);
    
            //calculate descriptors
            SiftDescriptorExtractor extractor;
            Mat descriptors;
            extractor.compute(image, keypoints, descriptors);
            
            write(database, "k"+std::to_string(i), keypoints);
            write(database, "d"+std::to_string(i), descriptors);
        }
        if (i % 100 == 0) std::cout << "Yey" << i << std::endl;
    }
    
    database.release();
    waitKey(0);
    return 0;
}
