#include "ncnn_mtcnn_tld_so.hpp"
#include "mtcnn.h"
#include "TLD.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
struct Bbox;
class Impl{
public:
	void InitMtcnn(const string& model_path, const int& minFace);
	void Detect(cv::Rect& result, cv::Mat& img);
	std::vector<Bbox> finalBbox;
	int skip=0;
	cv::Mat first;
	cv::Rect box;
	cv::Mat current_gray;
	cv::Rect pbox;
	bool status=true;
	cv::Mat last_gray;
	TLD tld;
	MTCNN mtcnn;
};
void Impl::InitMtcnn(const string& model_path, const int& minFace){
	mtcnn.init(model_path);
	mtcnn.SetMinFace(minFace);
}
void Impl::Detect(cv::Rect& result, cv::Mat& img){
	if (int(finalBbox.size())==0 || status==false){
		finalBbox.clear();
		mtcnn.detect(img,finalBbox);
		if(finalBbox.size()>0){
			box.x = finalBbox[0].x1;
			box.y = finalBbox[0].y1;
			box.width = finalBbox[0].x2-finalBbox[0].x1;
			box.height = finalBbox[0].y2-finalBbox[0].y1;
			cvtColor(img, last_gray, CV_RGB2GRAY);
			tld.defineLastBox(box);
			status=true;
		}
	}
	if(finalBbox.size()>0){
		cvtColor(img, current_gray, CV_BGR2GRAY);
		double t2 = (double)getTickCount();
		tld.processFrame(last_gray,current_gray,pbox,status);
		printf("tracking  %gms\n", ((double)getTickCount()-t2)*1000/getTickFrequency());
		if(status){
			pbox.width = box.width;
			pbox.height = box.height;
			if (skip>2){
				double t1 = (double)getTickCount();
				if (mtcnn.rnet(img, pbox)<0.99)finalBbox.clear();
				printf("rnet  %gms\n", ((double)getTickCount()-t1)*1000/getTickFrequency());
				skip=0;
			}
			result=pbox;
			swap(last_gray,current_gray);
		}
		skip++;
	}
}
faceTrack::faceTrack() : impl_(new Impl()){
	return;
}
void faceTrack::Init(const std::string& model_path, const int& minFace){
	Impl *p = (Impl *)impl_;
	p->InitMtcnn(model_path,minFace);
}
void faceTrack::DetectFace(cv::Rect& result, cv::Mat& img){
	Impl *p = (Impl *)impl_;
	p->Detect(result,img);
}
