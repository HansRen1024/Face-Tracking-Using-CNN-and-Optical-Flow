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
	std::string modelPath="./models";
	int minFace = 40;
	tracker.Init(modelPath, minFace);

//	std::string savePath = "./saved_2.avi";
//	cv::VideoWriter outputVideo;
//	outputVideo.open(savePath, CV_FOURCC('M', 'P', '4', '2'), 25.0, cv::Size(320,240));

	while (capture.read(frame)) {
		int q = cv::waitKey(1);
		if (q == 27) break;
		cv::Rect result;
		double t1 = (double)getTickCount();
		tracker.DetectFace(result, frame);
		printf("total %gms\n", ((double)getTickCount()-t1)*1000/getTickFrequency());
		printf("------------------\n");
		rectangle(frame,result,Scalar(0,0,255), 2);
		imshow("frame", frame);
//		outputVideo << frame;
	}
//	outputVideo.release();
	capture.release();
	cv::destroyAllWindows();
	return 0;
}
