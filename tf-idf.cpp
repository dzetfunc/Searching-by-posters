#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

const int num_clust = 1000;
const int num_img = 9702;

int main() {
    FileStorage clustbase("clust.yml", FileStorage::READ);
    FileStorage tf_idf("tfidf.yml", FileStorage::WRITE);
    
    vector<vector<float> > tf(num_img, vector<float> (num_clust, 0));
    
    vector<int> key_num;
    FileNode kek = clustbase["num"];
    read(kek, key_num);
    
    Mat clust_point;
    FileNode lel = clustbase["labels"];
    read(lel, clust_point);
    
    //подсчитываем кол-во точек в кластере на данном изображении
    int k = 0;
    for (int i = 0; i < num_img; ++i) {
        for (int j = 0; j < key_num[i]; ++j) {
            tf[i][clust_point.at<int>(k)]++;
            k++;
        }
    }
    
    //считаем IDF
    vector<float> idf(num_clust);
    for (int i = 0; i < num_clust; ++i) {
        float d = 0;
        for (int j = 0; j < num_img; ++j) {
            if (tf[j][i] != 0) d++;
        }
        idf[i] = log((float)num_img / d);
    }
    
    //считаем TF-IDF метрику
    for (int i = 0; i < num_img; ++i) {
        for (int j = 0; j < num_clust; ++j) {
            tf[i][j] = tf[i][j] * idf[j] / (float)key_num[i];
            k++;
        }
    }
    
    for (int i = 0; i < num_img; ++i) {
        write(tf_idf, "nyan" + std::to_string(i + 298), tf[i]);
    }
    
    write(tf_idf, "nyan_idf", idf);

    clustbase.release();
    tf_idf.release();
    
    cout << endl << "Y" << endl;
    
    waitKey(0);
    return 0;
}