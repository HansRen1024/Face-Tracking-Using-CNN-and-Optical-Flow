#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
class TLD{
private:
//Last frame data
  cv::Rect lastbox;
  //Tracker data
  bool tracked;
  cv::Rect tbb;
  vector<Point2f> points1;
  vector<Point2f> points2;
  void bbPoints(const cv::Rect& bb);
  void bbPredict(const cv::Rect& bb1,cv::Rect& bb2);
  std::vector<cv::Point2f> pointsFB;
  cv::Size window_size= Size(4,4);
  int level = 5;
  std::vector<uchar> status;
  std::vector<uchar> FB_status;
  std::vector<float> similarity;
  std::vector<float> FB_error;
  //  float simmed;
  float fbmed;
  cv::TermCriteria term_criteria= TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03);;
  float lambda = 0.5;
  void normCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
  bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
public:
  //Constructors
  TLD();
  //Methods
  void defineLastBox(const cv::Rect &box);
  void processFrame(const cv::Mat& img1,const cv::Mat& img2, cv::Rect& bbnext,bool& lastboxfound);
  void track(const cv::Mat& img1, const cv::Mat& img2);
  bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
};

