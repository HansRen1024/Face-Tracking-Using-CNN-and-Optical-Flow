#include "ncnn_mtcnn_tld_so.hpp"
#include "mtcnn.h"
#include "TLD.h"
#include "tld_utils.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
struct Bbox;
struct BoundingBox;
class Impl{
public:
	void globleInit(const string& model_path, const int& minFace);
	void Detect(cv::Rect& result, cv::Mat& img, int& detNumToTack);
	std::vector<Bbox> finalBbox;
	int skip=0;
	cv::Mat first;
	cv::Rect box;
	cv::Mat current_gray;
	BoundingBox pbox;
	vector<Point2f> pts1;
	vector<Point2f> pts2;
	bool status=true;
	cv::Mat last_gray;
	TLD tld;
	MTCNN mtcnn;
};
void Impl::globleInit(const string& model_path, const int& minFace){
	mtcnn.init(model_path);
	mtcnn.SetMinFace(minFace);
}
void Impl::Detect(cv::Rect& result, cv::Mat& img, int& detNumToTack){
	if (int(finalBbox.size())==0 || status==false){
		finalBbox.clear();
		mtcnn.detect(img,finalBbox);
		if (int(finalBbox.size())>0) skip++;
		else skip=0;
		if (skip>detNumToTack){
			tld.clearV();
			box.x = finalBbox[0].x1;
			box.y = finalBbox[0].y1;
			box.width = finalBbox[0].x2-finalBbox[0].x1;
			box.height = finalBbox[0].y2-finalBbox[0].y1;
			cvtColor(img, last_gray, CV_RGB2GRAY);
			tld.init(last_gray,box);
			status=true;
			skip=0;
		}
	}
	if (skip>detNumToTack){
		cvtColor(img, current_gray, CV_BGR2GRAY);
		tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,true);
		pbox.width = finalBbox[0].x2-finalBbox[0].x1;
		pbox.height =finalBbox[0].y2-finalBbox[0].y1;
		result=pbox;
		if (mtcnn.rnet(img, pbox)<0.98) finalBbox.clear();
		swap(last_gray,current_gray);
	}
	pts1.clear();
	pts2.clear();
	skip++;
}
faceTrack::faceTrack() : impl_(new Impl()){
	return;
}
void faceTrack::Init(const std::string& model_path, const int& minFace){
	Impl *p = (Impl *)impl_;
	p->globleInit(model_path,minFace);
}
void faceTrack::DetectFace(cv::Rect& result, cv::Mat& img, int& detNumToTack){
	Impl *p = (Impl *)impl_;
	p->Detect(result,img,detNumToTack);
}
