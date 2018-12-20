/*
 * ncnn_mtcnn_tld_so.hpp
 *
 *  Created on: Dec 7, 2018
 *      Author: hans
 */

#ifndef NCNN_MTCNN_TLD_SO_HPP_
#define NCNN_MTCNN_TLD_SO_HPP_

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
class faceTrack{
public:
	faceTrack();
	void Init(const std::string& model_path, const int& minFace);
	void DetectFace(cv::Rect& result, cv::Mat& img);
private:
    void* impl_;
};



#endif /* NCNN_MTCNN_TLD_SO_HPP_ */
