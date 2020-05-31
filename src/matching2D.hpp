#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include "chebyshev.h"
#include "eggplot.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

void fillCombination(std::vector<Results> &combination);
void generateReport(std::vector<Results> &combinations,std::ofstream &outputFile);
void plot(std::vector<Results> &combinations);
double calculateRMSE(std::vector<double> &gt , std::vector<double> &output);
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t_des, bool bVis=false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t_des, bool bVis=false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, 
                        std::string detectorType, double& t_des, bool bVis=false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, 
                   cv::Mat &descriptors, std::string descriptorType,double& t_des);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
                     cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches,
                       std::string descriptorType, std::string matcherType, std::string selectorType,double& t_des);

#endif /* matching2D_hpp */
