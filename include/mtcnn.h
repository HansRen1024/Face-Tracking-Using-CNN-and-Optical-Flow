#pragma once
#ifndef __MTCNN_NCNN_H__
#define __MTCNN_NCNN_H__
#include "net.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <map>
#include <iomanip>
#include <sstream>
#include <iostream>
using namespace std;
using namespace cv;
struct Bbox {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    float ppoint[10];
    float regreCoord[4];
};
bool cmpScore(Bbox lsh, Bbox rsh) {
	if (lsh.score < rsh.score) return true;
	else return false;
}
bool cmpArea(Bbox lsh, Bbox rsh) {
	if (lsh.area < rsh.area) return false;
	else return true;
}
bool sortScore(Bbox& lsh, Bbox& rsh){
	return lsh.score>rsh.score;
}
static float getElapse(struct timeval *tv1,struct timeval *tv2){
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}
class MTCNN {
public:
	MTCNN();
	MTCNN(const string &model_path);
    ~MTCNN();
    void init(const string &model_path);
	void SetMinFace(int minSize);
	float rnet(cv::Mat& image, cv::Rect& face_);
    int detectMain(cv::Mat& cv_img, vector<int>& faceBox, float& costTime);
    void detect(cv::Mat& cv_img, std::vector<Bbox>& finalBbox);
private:
    float calIOU(Bbox &box1,Bbox &box2,const string& modelname);
    float rou(float src, int bits);
    void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, float scale);
    void nms(vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname="Union");
    void refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square);
    void PNet();
    void RNet();
    void ONet();
    ncnn::Net Pnet, Rnet, Onet;
    ncnn::Mat img;
    const float nms_threshold[3] = {0.5f, 0.7f, 0.7f};
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
	const int MIN_DET_SIZE = 10;
	std::vector<Bbox> firstPreviousBbox_, secondPreviousBbox_, thirdPrevioussBbox_;
    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    int img_w=0;
    int img_h=0;
    vector<Bbox> rectangles;
    Bbox LaFaceBox,CuFaceBox,rectangle,MeanNose,MeanFaceBox;
    int pad=0;
    vector< pair<int,int>> NoseList,CuFaceBoxList;
    std::vector<int> padList;
    float IOUthres = 0.90;
	int MeanFrame = 5;
	string MinIOU="Min";
	string UnionIOU="Union";
private:
	const float threshold[3] = { 0.8f, 0.8f, 0.6f };
	int minsize = 40;
	const float pre_facetor = 0.709f;
};
MTCNN::MTCNN(){
}
MTCNN::MTCNN(const string &model_path) {
	std::vector<std::string> param_files = {
		model_path+"/det1.param",
		model_path+"/det2.param",
		model_path+"/det3.param"
	};
	std::vector<std::string> bin_files = {
		model_path+"/det1.bin",
		model_path+"/det2.bin",
		model_path+"/det3.bin"
	};
	Pnet.load_param(param_files[0].data());
	Pnet.load_model(bin_files[0].data());
	Rnet.load_param(param_files[1].data());
	Rnet.load_model(bin_files[1].data());
	Onet.load_param(param_files[2].data());
	Onet.load_model(bin_files[2].data());
}
MTCNN::~MTCNN(){
    Pnet.clear();
    Rnet.clear();
    Onet.clear();
}
void MTCNN::SetMinFace(int minSize){
	minsize = minSize;
}
void MTCNN::init(const string &model_path){
	std::vector<std::string> param_files = {
		model_path+"/det1.param",
		model_path+"/det2.param",
		model_path+"/det3.param"
	};
	std::vector<std::string> bin_files = {
		model_path+"/det1.bin",
		model_path+"/det2.bin",
		model_path+"/det3.bin"
	};
	Pnet.load_param(param_files[0].data());
	Pnet.load_model(bin_files[0].data());
	Rnet.load_param(param_files[1].data());
	Rnet.load_model(bin_files[1].data());
	Onet.load_param(param_files[2].data());
	Onet.load_model(bin_files[2].data());
}
float MTCNN::calIOU(Bbox &box1,Bbox &box2,const string& modelname){
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	float startx = 0;
	float endx = 0;
	float starty = 0;
	float endy = 0;
	float width = 0;
	float height = 0;
	endx = (box1.x2>box2.x2)?box1.x2:box2.x2;
	startx = (box1.x1>box2.x1)?box1.x1:box2.x1;
	endy = (box1.y2>box2.y2)?box1.y2:box2.y2;
	starty = (box1.y1>box2.y1)?box1.y1:box2.y1;
	width = (box1.x2-box1.x1)+(box2.x2-box2.x1)-(endx-startx);
	height = (box1.y2-box1.y1)+(box2.y2-box2.y1)-(endy-starty);
	if (width>0 and height>0){
		maxX = startx;
		maxY = starty;
		minX = (box1.x2<box2.x2)?box1.x2:box2.x2;
		minY = (box1.y2<box2.y2)?box1.y2:box2.y2;
		if(!modelname.compare("Union")){
			maxX = (minX+1>maxX)?(minX-maxX+1):(maxX-minX+1);
			maxY = (minY+1>maxY)?(minY-maxY+1):(maxY-minY+1);
		}
		else{
			maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
			maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
		}
		IOU = maxX * maxY;
		if(!modelname.compare("Union"))IOU = IOU/(box1.area + box2.area - IOU);
		else if(!modelname.compare("Min"))IOU = IOU/((box1.area<box2.area)?box1.area:box2.area);
	}
	return IOU;
}
void MTCNN::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, float scale){
    const int stride = 2;
    const int cellsize = 12;
    float *p = score.channel(1);//score.data + score.cstep;
    Bbox bbox;
    float inv_scale = 1.0f/scale;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;
                bbox.x1 = round((stride*col+1)*inv_scale);
                bbox.y1 = round((stride*row+1)*inv_scale);
                bbox.x2 = round((stride*col+1+cellsize)*inv_scale);
                bbox.y2 = round((stride*row+1+cellsize)*inv_scale);
                bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
                const int index = row * score.w + col;
                for(int channel=0;channel<4;channel++){
                    bbox.regreCoord[channel]=location.channel(channel)[index];
                }
                boundingBox_.push_back(bbox);
            }
            p++;
        }
    }
}
void MTCNN::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty())return;
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    std::vector<int> vPick;
    int nPick = 0;
    std::multimap<float, int> vScores;
    const int num_boxes = boundingBox_.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i){
		vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
	}
    while(vScores.size() > 0){
        int last = vScores.rbegin()->second;
        vPick[nPick] = last;
        nPick += 1;
        for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
            int it_idx = it->second;
            maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //maxX1 and maxY1 reuse
            maxX = ((minX-maxX+1)>0)? (minX-maxX+1) : 0;
            maxY = ((minY-maxY+1)>0)? (minY-maxY+1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            if(!modelname.compare("Union"))
                IOU = IOU/(boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
            else if(!modelname.compare("Min")){
                IOU = IOU/((boundingBox_.at(it_idx).area < boundingBox_.at(last).area)? boundingBox_.at(it_idx).area : boundingBox_.at(last).area);
            }
            if(IOU > overlap_threshold){
                it = vScores.erase(it);
            }else{
                it++;
            }
        }
    }
    vPick.resize(nPick);
    std::vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++){
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}
void MTCNN::refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        bbw = (*it).x2 - (*it).x1 + 1;
        bbh = (*it).y2 - (*it).y1 + 1;
        x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
        y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
        x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
        y2 = (*it).y2 + (*it).regreCoord[3]*bbh;
        if(square){
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);
        }
        if((*it).x1<0)(*it).x1=0;
        if((*it).y1<0)(*it).y1=0;
        if((*it).x2>width)(*it).x2 = width - 1;
        if((*it).y2>height)(*it).y2 = height - 1;
        it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
    }
}
void MTCNN::PNet(){
    firstBbox_.clear();
    float minl = img_w < img_h? img_w: img_h;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = pre_facetor;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
        scales_.push_back(m);
        minl *= factor;
        m = m*factor;
    }
    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        ncnn::Mat in;
        resize_bilinear(img, in, ws, hs);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_num_threads(4);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        std::vector<Bbox> boundingBox_;
        generateBbox(score_, location_, boundingBox_, scales_[i]);
        nms(boundingBox_, nms_threshold[0]);
        firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
        boundingBox_.clear();
    }
}
void MTCNN::RNet(){
    secondBbox_.clear();
    for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 24, 24);
        ncnn::Extractor ex = Rnet.create_extractor();
		ex.set_num_threads(4);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox;
        ex.extract("prob1", score);
        ex.extract("conv5-2", bbox);
		if ((float)score[1] > threshold[1]) {
			for (int channel = 0; channel<4; channel++) {
				it->regreCoord[channel] = (float)bbox[channel];//*(bbox.data+channel*bbox.cstep);
			}
			it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
			it->score = score.channel(1)[0];//*(score.data+score.cstep);
			secondBbox_.push_back(*it);
		}
    }
}
float MTCNN::rnet(cv::Mat& image, cv::Rect& face_){
	Bbox face;
	face.x1=face_.x;
	face.y1=face_.y;
	face.x2=face_.x+face_.width;
	face.y2=face_.y+face_.height;
	ncnn::Mat img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	img_w = img.w;
	img_h = img.h;
	ncnn::Mat tempIm;
	copy_cut_border(img, tempIm, face.y1, img_h-face.y2, face.x1, img_w-face.x2);
	ncnn::Mat in;
	resize_bilinear(tempIm, in, 24, 24);
	in.substract_mean_normalize(mean_vals, norm_vals);
	ncnn::Extractor ex = Rnet.create_extractor();
	ex.set_num_threads(4);
	ex.set_light_mode(true);
	ex.input("data", in);
	ncnn::Mat score;
	ex.extract("prob1", score);
	return (float)score[1];
}
void MTCNN::ONet(){
    thirdBbox_.clear();
    for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 48, 48);
        ncnn::Extractor ex = Onet.create_extractor();
		ex.set_num_threads(4);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox, keyPoint;
        ex.extract("prob1", score);
        ex.extract("conv6-2", bbox);
        ex.extract("conv6-3", keyPoint);
		if ((float)score[1] > threshold[2]) {
			for (int channel = 0; channel < 4; channel++) {
				it->regreCoord[channel] = (float)bbox[channel];
			}
			it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
			it->score = score.channel(1)[0];
			for (int num = 0; num<5; num++) {
				(it->ppoint)[num] = it->x1 + (it->x2 - it->x1) * keyPoint[num];
				(it->ppoint)[num + 5] = it->y1 + (it->y2 - it->y1) * keyPoint[num + 5];
			}
			thirdBbox_.push_back(*it);
		}
    }
}
void MTCNN::detect(cv::Mat& cv_img, std::vector<Bbox>& finalBbox_){
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);
    img = ncnn_img;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);
    PNet();
    if(firstBbox_.size() < 1) return;
    nms(firstBbox_, nms_threshold[0]);
    refine(firstBbox_, img_h, img_w, true);
    RNet();
    if(secondBbox_.size() < 1) return;
    nms(secondBbox_, nms_threshold[1]);
    refine(secondBbox_, img_h, img_w, true);
    ONet();
    if(thirdBbox_.size() < 1) return;
    refine(thirdBbox_, img_h, img_w, true);
    nms(thirdBbox_, nms_threshold[2], "Min");
    finalBbox_ = thirdBbox_;
    if (int(finalBbox_.size()>0)) sort(finalBbox_.begin(),finalBbox_.end(),sortScore);
}
float MTCNN::rou(float src, int bits){
	stringstream ss;
	ss << fixed << setprecision(bits) << src;
	ss >> src;
	return src;
}
int MTCNN::detectMain(cv::Mat& cv_img, vector<int>& faceBox, float& costTime){
	faceBox.clear();
	rectangles.clear();
	struct timeval  tv1,tv2;
	struct timezone tz1,tz2;
	gettimeofday(&tv1,&tz1);
	detect(cv_img, rectangles);
	if (int(rectangles.size())==0){
		LaFaceBox.exist=false;
		return 0;
	}
	rectangle = rectangles[0];
	float rouRate  = rou((rectangle.x2-rectangle.x1)/float(cv_img.cols),1);
	pad = round(1.0*rouRate*cv_img.cols/2);
//	pad = round((rectangle.x2-rectangle.x1)/2);
	padList.push_back(pad);
	NoseList.push_back(make_pair(round((rectangle.x2+rectangle.x1)/2),round((rectangle.y2+rectangle.y1)/2)));
//	NoseList.push_back(make_pair(rectangle.ppoint[2],rectangle.ppoint[7]));
	while (int(NoseList.size())>MeanFrame){
		NoseList.erase(NoseList.begin());
		padList.erase(padList.begin());
	}
	int MeanPad=0;
	for(vector<int>::iterator it=padList.begin(); it!=padList.end();it++){
		MeanPad+=int(round((*it)/float(padList.size())));
	}
	padList.push_back(MeanPad);
	MeanNose.x1=0;
	MeanNose.y1=0;
	for(vector< pair<int,int>>::iterator it=NoseList.begin(); it!=NoseList.end();it++){
		MeanNose.x1+=int(round((*it).first/float(NoseList.size())));
		MeanNose.y1+=int(round((*it).second/float(NoseList.size())));
	}
	NoseList.push_back(make_pair(MeanNose.x1,MeanNose.y1));
	CuFaceBox.x1 = MeanNose.x1-MeanPad;
	CuFaceBox.y1 = MeanNose.y1-MeanPad;
	CuFaceBox.x2 = MeanNose.x1+MeanPad;
	CuFaceBox.y2 = MeanNose.y1+MeanPad;
	if(!LaFaceBox.exist){
		LaFaceBox=CuFaceBox;
		LaFaceBox.exist=true;
	}
	LaFaceBox.area = (LaFaceBox.x2-LaFaceBox.x1)*(LaFaceBox.y2-LaFaceBox.y1);
	CuFaceBox.area = (CuFaceBox.x2-CuFaceBox.x1)*(CuFaceBox.y2-CuFaceBox.y1);
	float IOUrate = calIOU(LaFaceBox,CuFaceBox,UnionIOU);
	if (IOUrate>IOUthres)CuFaceBox=LaFaceBox;
	LaFaceBox=CuFaceBox;
	CuFaceBoxList.push_back(make_pair(CuFaceBox.x1,CuFaceBox.y1));
	while (int(CuFaceBoxList.size())>MeanFrame)CuFaceBoxList.erase(CuFaceBoxList.begin());
	MeanFaceBox.x1=0;
	MeanFaceBox.y1=0;
	for(vector< pair<int,int>>::iterator it=CuFaceBoxList.begin(); it!=CuFaceBoxList.end();it++){
		MeanFaceBox.x1+=int(round((*it).first/float(CuFaceBoxList.size())));
		MeanFaceBox.y1+=int(round((*it).second/float(CuFaceBoxList.size())));
	}
	CuFaceBoxList.push_back(make_pair(MeanFaceBox.x1,MeanFaceBox.y1));
	faceBox.resize(4);
	faceBox[0] = MeanFaceBox.x1;
	faceBox[1] = MeanFaceBox.y1;
	faceBox[2] = MeanFaceBox.x1+MeanPad*2;
	faceBox[3] = MeanFaceBox.y1+MeanPad*2;
	gettimeofday(&tv2,&tz2);
	costTime = getElapse(&tv1, &tv2);
	return 0;
}
#endif //__MTCNN_NCNN_H__
