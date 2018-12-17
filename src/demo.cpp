#include <opencv2/opencv.hpp>
#include "ncnn_mtcnn_tld_so.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

int main(){
	cv::VideoCapture capture(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH,320);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
	if (!capture.isOpened()) return -1;
	cv::Mat frame;
	faceTrack tracker;
	std::string modelPath="../models";
	tracker.Init(modelPath, 40);
	int detNumToTack=0;
	while (capture.read(frame)) {
		int q = cv::waitKey(1);
		if (q == 27) break;
		cv::Rect result;
		double t1 = (double)getTickCount();
		tracker.DetectFace(result, frame, detNumToTack);
		printf(" %gms\n", ((double)getTickCount()-t1)*1000/getTickFrequency());
		rectangle(frame,result,Scalar(0,0,255), 2);
		imshow("frame", frame);
	}
	capture.release();
	cv::destroyAllWindows();
	return 0;
}
