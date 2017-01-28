#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <string>

using namespace cv;
using namespace std;

const int num_clust = 1000;
const int num_img = 9702;

int main() {
    FileStorage tf_idf("tfidf.yml", FileStorage::READ);
    FileStorage inv("invert.yml", FileStorage::WRITE);
    
    //считаем инвертированный индекс
    vector<vector<int > > invert(num_clust);
    
    for (int i=298; i < 298 + num_img; ++i) {
        
        vector<float> tfidf;
        FileNode lel = tf_idf["nyan"+to_string(i)];
        read(lel, tfidf);
        
        for (int j = 0; j < num_clust; ++j) {
            if (tfidf[j] != 0) {
                invert[j].push_back(i);
            }
        }
    }
    
    for (int i = 0; i < num_clust; ++i) {
        write(inv, "use" + to_string(i), invert[i]);
    }
    
    tf_idf.release();
    inv.release();

    return 0;
}