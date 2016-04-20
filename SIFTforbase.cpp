#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <string>

using namespace cv;


int main() {
    //записываем все в 5 файлов
    FileStorage database("base8000-9999.yml", FileStorage::WRITE);
    
    for (int i = 8000; i < 10000; ++i) {
        Mat image;
        image = imread("../../../Downloads/kinopoisk/" + std::to_string(i) + ".jpg");
        
        if (image.data) {
            //находим особые точки
            SurfFeatureDetector detector(4800);
    
            std::vector<KeyPoint> keypoints;
            detector.detect(image, keypoints);
    
            //считаем их дескрипторы
            SurfDescriptorExtractor extractor;
            Mat descriptors;
            extractor.compute(image, keypoints, descriptors);
            
            write(database, "d" + std::to_string(i), descriptors);
        }
    }
    
    database.release();
    waitKey(0);
    return 0;
}
