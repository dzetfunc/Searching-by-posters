#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <string>

using namespace cv;
using namespace std;


int main() {
    //записываем все в 5 файлов
    FileStorage database("base8000-9999.yml", FileStorage::WRITE);
    
    for (int i = 8000; i < 10000; ++i) {
        Mat image;
        image = imread("../../../Downloads/kinopoisk/" + to_string(i) + ".jpg");
        
        if (image.data) {
            resize(image, image, Size(480, 640), 0, 0, INTER_LINEAR);
            
            GaussianBlur(image, image, Size(5,5), 1.2, 0, BORDER_DEFAULT );
            
            //находим особые точки
            SiftFeatureDetector detector(0, 3, 0.18, 0);
    
            vector<KeyPoint> keypoints;
            detector.detect(image, keypoints);
    
            //считаем их дескрипторы
            SiftDescriptorExtractor extractor;
            Mat descriptors;
            extractor.compute(image, keypoints, descriptors);
            
            write(database, "d" + std::to_string(i), descriptors);
        }
        if ((i+1) % 100 == 0) cout << endl << endl << "YEY" + to_string(i) << endl << endl;
    }
    
    database.release();
    return 0;
}
