#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

const int num_clust = 1000;
const int num_img = 9702;

bool comp( pair<float, int> a, pair<float, int> b) {
    return (a.first < b.first);
}

int main() {
    Mat image;
    image = imread("new.jpg");
    
    resize(image, image, Size(480, 640), 0, 0, INTER_LINEAR);
    
    GaussianBlur(image, image, Size(5,5), 1.2, 0, BORDER_DEFAULT );
    
    //находим особые точки и дескрипторы поданного на вход изображения
    std::vector<KeyPoint> keypoints;
    
    SiftFeatureDetector detector(0, 3, 0.18, 0);
    detector.detect(image, keypoints);

    SiftDescriptorExtractor extractor;
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
        for (int j = 0; j < 128; ++j) {
            min += (descriptors.at<float>(i, j) - centers.at<float>(0, j)) * (descriptors.at<float>(i, j) - centers.at<float>(0, j));
        }
        for (int t = 1; t < num_clust; ++t) {
            float x = 0;
            for (int j = 0; j < 128; ++j) {
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
    
    //находим минимальнst отличающиеся tf-idf метрики
    vector<pair<float, int> > d;
    
    for (int i = 298; i < 298 + num_img; ++i) {
        if (marks[i]) {
            d.push_back(make_pair(0, i));
            vector<float> tf;
            FileNode sas = tfbase["nyan" + to_string(i)];
            read(sas, tf);
            for (int j = 0; j < num_clust; ++j) {
                d[d.size() - 1].first += (clus[j] - tf[j]) * (clus[j] - tf[j]);
            }
        }
    }
    
    sort(d.begin(), d.end(), comp);
    
    vector <int> same;
    for (int i = 0; i < 100; i++) {
        same.push_back(d[i].second);
    }
    
    tfbase.release();
    
    int max = 0;
    int maxx = same[0];
    vector <int> c(100, 0);

    for (int i = 0; i < 100; i++) {
        if (same[i] != 0) {
            Mat input_im;
            input_im = imread("../../../Downloads/kinopoisk/" + to_string(same[i]) + ".jpg");
            
            resize(image, image, Size(480, 640), 0, 0, INTER_LINEAR);
            
            GaussianBlur(input_im, input_im, Size(5,5), 1.2, 0, BORDER_DEFAULT );
            
            SiftFeatureDetector detector(0, 3, 0.18, 0);
        
            vector<KeyPoint> keypoints_input;
            detector.detect(input_im, keypoints_input);
        
            //считаем их дескрипторы
            SiftDescriptorExtractor extractor;
            Mat descriptors_input;
            extractor.compute(input_im, keypoints_input, descriptors_input);

            //matching descriptor
            FlannBasedMatcher matcher;
            vector<DMatch> matches;
            if (descriptors_input.type() == 0) continue;
            matcher.match(descriptors, descriptors_input, matches);

            //calculate distance
            double max_dist = 0; double min_dist = 100;
            for (int i = 0; i < descriptors.rows; ++i)
            { double dist = matches[i].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }


            vector<DMatch> good_matches;
            for( int i = 0; i < descriptors.rows; i++ ) {
                if (matches[i].distance <= 3*min_dist)
                    good_matches.push_back(matches[i]);
            }
        
      
            vector<Point2f> im;
            vector<Point2f> im_input;
        
            for( int i = 0; i < good_matches.size(); i++ )
            {
                //точки хороших матчей
                im.push_back( keypoints[ good_matches[i].queryIdx ].pt );
                im_input.push_back( keypoints_input[ good_matches[i].trainIdx ].pt );
            }
      
            Mat mask, M;
            
            if (im.size() >= 4) {
                M = findHomography(im, im_input, CV_RANSAC, 3, mask);
        
            int count = 0;

            for (int i = 0; i < mask.rows; i++)
                count += (int)mask.at<uchar>(i);
           if (max < count) {
                max = count;
                maxx = same[i];
            }
            c[i] = count;
            }
        }
    }
 
 

    for (int i = 0; i < 5; i++) {
        cout << endl << endl << "maybe:   http://www.kinopoisk.ru/film/" + to_string(same[i]) + "/" << to_string(c[i]) << endl << endl;
    }
    
    cout << endl << endl << "http://www.kinopoisk.ru/film/" + to_string(maxx) + "/" << to_string(max) << endl << endl;

    return 0;
}
