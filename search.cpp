#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>

using namespace cv;
using namespace std;

const int num_clust = 1000;
const int num_img = 9702;

int main() {
    Mat image;
    image = imread("image.jpg");
    //находим особые точки и дескрипторы поданного на вход изображения
    std::vector<KeyPoint> keypoints;
    
    SurfFeatureDetector detector(4800);
    detector.detect(image, keypoints);

    SurfDescriptorExtractor extractor;
    Mat descriptors;
    extractor.compute(image, keypoints, descriptors);
    
    FileStorage clustbase("clust.yml", FileStorage::READ);
    Mat centers;
    FileNode lel = clustbase["centers"];
    read(lel, centers);
    
    //кластеризуем дескрипторы поданного изображения
    vector<float> clus(num_clust, 0);
    
    for (int i = 0; i < (int)keypoints.size(); ++i){
        float min = 0;
        int minx = 0;
        for (int j = 0; j < 64; ++j) {
            min += (descriptors.at<float>(i, j) - centers.at<float>(0, j)) * (descriptors.at<float>(i, j) - centers.at<float>(0, j));
        }
        for (int t = 1; t < num_clust; ++t) {
            float x = 0;
            for (int j = 0; j < 64; ++j) {
                x += (descriptors.at<float>(i, j) - centers.at<float>(t, j)) * (descriptors.at<float>(i, j) - centers.at<float>(t, j));
            }
            if (x < min) {
                min = x;
                minx = t;
            }
        }
        clus[minx]++;
    }
    clustbase.release();
    
    FileStorage tfbase("tfidf.yml", FileStorage::READ);
    vector<float> idf;
    FileNode kek = tfbase["nyan_idf"];
    read(kek, idf);
    
    FileStorage invert("invert.yml", FileStorage::READ);
    
    //рассматриваем изображения, где есть хоть один кластер с поданного на вход
    vector<bool> marks(298 + num_img, false);
    for (int i = 0; i < num_clust; ++i) {
        clus[i] = clus[i] * (float)idf[i] / (float)keypoints.size();
        if (clus[i] != 0) {
            vector<float> inv;
            FileNode mem = invert["use"+to_string(i)];
            read(mem, inv);
            for (int j = 0; j < (int)inv.size(); ++j) marks[inv[j]] = true;
        }
    }
    
    invert.release();
    
    //находим минимальную отличающуюся tf-idf метрику
    float min = 0;
    int minx = 298;
    vector<float> tf;
    FileNode sas = tfbase["nyan298"];
    read(sas, tf);
    for (int j = 0; j < num_clust; ++j) {
        min += (clus[j] - tf[j]) * (clus[j] - tf[j]);
    }
    
    
    for (int i = 299; i < 298 + num_img; ++i) {
        if (marks[i]) {
            float d = 0;
            vector<float> tf;
            FileNode sas = tfbase["nyan" + to_string(i)];
            read(sas, tf);
            for (int j = 0; j < num_clust; ++j) {
                d += (clus[j] - tf[j]) * (clus[j] - tf[j]);
            }
            if (d < min) {
                min = d;
                minx = i;
            }
        }
    }
    
    tfbase.release();
    
    cout << endl << endl << minx << endl << endl;
    
    waitKey(0);
    return 0;
}
