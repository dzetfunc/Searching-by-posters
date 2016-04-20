#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <string>

using namespace cv;

const int num_clust = 1000;

int main() {
    FileStorage clustbase("clust.yml", FileStorage::WRITE);
    
    Mat samples, labels, centers;
    vector<int> keypoints_number;
    
    //пушим все вектора дескрипторов(sаmple) в общую матрицу
    FileStorage database("base298-1999.yml", FileStorage::READ);
    for(int i = 298; i < 2000; ++i) {
        Mat sample;
        FileNode kek = database["d" + std::to_string(i)];
        read(kek, sample);
        samples.push_back(sample);
        keypoints_number.push_back(sample.rows);
    }
    database.release();

    for(int i = 1; i < 5; ++i) {
        FileStorage database("base" + std::to_string(2000 * i)+ "-" + std::to_string(2000 * i + 1999) + ".yml", FileStorage::READ);
        for(int j = 2000 * i; j < 2000 * (i + 1); ++j) {
            Mat sample;
            FileNode kek = database["d" + std::to_string(j)];
            read(kek, sample);
            samples.push_back(sample);
            keypoints_number.push_back(sample.rows);
        }
        database.release();
    }
    
    //сохраняем вектор кол-ва особых точек для каждого изображения
    write(clustbase, "num", keypoints_number);
    
    //меняем тип матрицы на нужный
    Mat samples1;
    samples.convertTo(samples1, CV_32F, 1, 0);
    
    //кластеризуем
    double compact;
    compact = kmeans(samples1, num_clust, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
    
    
    write(clustbase, "labels", labels);
    write(clustbase, "centers", centers);
    
    
    std::cout << std::endl << std::endl << compact << std::endl << std::endl;
    

    clustbase.release();

    waitKey(0);
    return 0;
}