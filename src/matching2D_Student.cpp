#include <numeric>
#include "matching2D.hpp"

using namespace std;
void fillCombination(std::vector<Results> &combinations)
{

    vector<std::string> detectors    = { "HARRIS", "SHITOMASI", "FAST", "BRISK", "ORB" , "SIFT","AKAZE"};
    vector<std::string> descriptors  = { "BRISK", "BRIEF", "ORB", "FREAK", "SIFT","AKAZE"};                      
    // vector<std::string> detectors    = { "FAST"};
    // vector<std::string> descriptors  = { "BRIEF"}; 
    vector<std::string> matchers     = { "MAT_BF"};
    vector<std::string> selectors    = { "SEL_KNN"};
    for(auto detector : detectors)
    {
        for(auto descriptor : descriptors)
        {
            for(auto matcher : matchers)
            {
                for(auto selector : selectors)
                {
                    struct Results c;
                    c.detector = detector;
                    c.descriptor = descriptor;
                    c.matcher = matcher;
                    c.selector = selector;
                    c.detector_time_ms = 0.0f;
                    c.descriptor_time_ms =0.0f;
                    c.matcher_time_ms = 0.0f;
                    c.no_detected_points =0;
                    c.no_detected_points_on_vehicle =0;
                    c.no_descriptors = 0;
                    c.no_matched_points = 0;
                    c.no_images = 10;
                    combinations.push_back(c);
                }
            }

        }
    }
}

void generateReport(std::vector<Results> &combinations,std::ofstream &outputFile)
{
    std::cout<<"generate reports"<<std::endl;
    // create and open the .csv file
    outputFile << "Detector/Descriptor Type"
               << ","
               << "# KeyPoints"
               << ","
               << "#KeyPointsPerROI"
               << ","
               << "# Descriptors"
               << ","
               << "#MatchedPoints"
               << ","
               << "Average Detector+DescriptorTime(ms)"
               << ","
               << "Average overall(ms)"
               << ","
               << "RMSE(ms)"
               << std::endl;

    for( auto& combination : combinations )
    {
        double detect_time   = (combination.detector_time_ms);
        double desc_time     = (combination.descriptor_time_ms);
        double match_time    = (combination.matcher_time_ms);
        double detect_desc   = (detect_time + desc_time)/10.0;
        double avg           = (detect_time + desc_time + match_time)/10.0;
        std:: string det_des = combination.detector+"/"+combination.descriptor;
        double rmse = calculateRMSE(combination.lidar_ttc,combination.camera_ttc);

        outputFile  << det_des 
                    << "," << combination.no_detected_points
                    << "," << combination.no_detected_points_on_vehicle
                    << "," << combination.no_descriptors
                    << "," << combination.no_matched_points
                    << "," << std::fixed << std::setprecision(8) <<detect_desc
                    << "," << std::fixed << std::setprecision(8) <<avg
                    << "," << std::fixed << std::setprecision(8) <<rmse
                    << endl;
    }
    outputFile.close();
}

void plot(std::vector<Results> &combinations)
{
    for( auto& combination : combinations )
    {
        // Show on screen and
        // export to .png, .eps, .pdf, .html, .svg files
        eggp::Eggplot plt(eggp::PNG);
        std::string title = combination.detector+"/"+combination.descriptor;
        std::string out = combination.detector+"_"+combination.descriptor;
        plt.title(title);
        plt.plot({combination.avg_time_ms,combination.camera_ttc,combination.avg_time_ms,combination.lidar_ttc});
        plt.xlabel("time");
        plt.ylabel("TTC ms");
        plt.linespec(1, {{eggp::MarkerSize, "0.5"}, {eggp::Marker, "*"}});
        plt.linespec(2, {{eggp::MarkerSize, "0.5"}, {eggp::Marker, "."}});
        plt.linespec(1, eggp::Color, "b");
        plt.linespec(2, eggp::Color, "g");
        plt.legend({title, "Lidar"});
        plt.grid(true);
        plt.print(out);
        plt.exec(); 
    }
}

double calculateRMSE(std::vector<double> &gt , std::vector<double> &output)
{
    double root=0.0,mean=0.0,square=0.0;
    if(gt.size()!= output.size() || output.size()==0)
    {
        std::cout<<"The sizes are not correct"<<std::endl;
        std::cout<<"gt size"<<gt.size()<<std::endl;
        std::cout<<"output size"<<output.size()<<std::endl;
        root =0.0;
    }
    else
    {
        //calculate the square error
        for(int i ; i<gt.size();i++)
        {
            std::cout << "gt vak" << gt.at(i) << std::endl;
            std::cout << "est val" << output.at(i) << std::endl;
            
            double diff = gt.at(i) - output.at(i);
            std::cout << "diff" << diff << std::endl;

            square+= pow(diff,2);
        }

        //calculate the mean
        mean = square/(float)(gt.size());

        //calculate root
        root = sqrt(mean);
    }
    return root;
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t_des, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);


    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    // Apply corner detection
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {   // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    } // eof loop over rows
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t_des = 1000 * t / 1.0;
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t_des, bool bVis)
{
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);
    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t_des = 1000 * t / 1.0;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Modern detector
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        std::string detectorType,double& t_des,  bool bVis)
{
    // Select modern detector based on input
    string windowName;
    cv::Ptr<cv::FeatureDetector> detector; 
    if(detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if(detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cout<<"The detector type "<<detectorType<<"is not avalible in list"<<std::endl;
        std::cout<<"The default detector fast will be used"<<endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
    }
    

    // Apply corner detection
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t_des = 1000 * t / 1.0;
    cout << detectorType<<" "<<"detection with n=" << keypoints.size() << 
           " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = " Corner Detector Results";
        windowName = detectorType + windowName;
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType,
                   double& t_des)
{
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t_des = 1000 * t / 1.0;
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches,
                      std::string descriptorType, std::string matcherType, std::string selectorType, double& t_des)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    double t;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {   // OpenCV bug workaround : convert binary descriptors
            // to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "FLANN matching";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        t = (double)cv::getTickCount();

        // Finds the best match for each descriptor in desc1
        matcher->match(descSource, descRef, matches); 

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches

        // STUDENT TASK
        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        t_des = 1000 * t / 1.0;
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
        // EOF STUDENT TASK
    }
}