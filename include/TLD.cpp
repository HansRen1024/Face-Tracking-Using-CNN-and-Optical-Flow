#include "TLD.h"
using namespace cv;
using namespace std;
float median(vector<float> v){
    int n = floor(v.size() / 2);
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}
void TLD::defineLastBox(const Rect& box){
  lastbox=box;
}
TLD::TLD(){}
void TLD::processFrame(const cv::Mat& img1,const cv::Mat& img2,cv::Rect& bbnext,bool& lastboxfound){
  points1.clear();
  points2.clear();
  if(lastboxfound) track(img1,img2);
  else tracked = false;
  if (tracked){
	  bbnext=tbb;
	  lastbox=bbnext;
  }
  else lastboxfound = false;
}
void TLD::track(const Mat& img1, const Mat& img2){
  bbPoints(lastbox);
  if (points1.size()<1){
      tracked=false;
      return;
  }
  tracked = trackf2f(img1,img2,points1,points2);
  if (tracked)bbPredict(lastbox,tbb);
}
void TLD::bbPoints(const cv::Rect& bb){
  int max_pts=10;
  int margin_h=0;
  int margin_v=0;
  int stepx = ceil((bb.width-2*margin_h)/max_pts);
  int stepy = ceil((bb.height-2*margin_v)/max_pts);
  for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
    	  points1.push_back(Point2f(x,y));
      }
  }
}
void TLD::bbPredict(const cv::Rect& bb1,cv::Rect& bb2){
  int npoints = (int)points1.size();
  vector<float> xoff(npoints);
  vector<float> yoff(npoints);
  for (int i=0;i<npoints;i++){
      xoff[i]=points2[i].x-points1[i].x;
      yoff[i]=points2[i].y-points1[i].y;
  }
  float dx = median(xoff);
  float dy = median(yoff);
  float s;
  if (npoints>1){
      vector<float> d;
      d.reserve(npoints*(npoints-1)/2);
      for (int i=0;i<npoints;i++){
          for (int j=i+1;j<npoints;j++)
            d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
      }
      s = median(d);
  }
  else s = 1.0;
  float s1 = 0.5*(s-1)*bb1.width;
  float s2 = 0.5*(s-1)*bb1.height;
  bb2.x = round( bb1.x + dx -s1);
  bb2.y = round( bb1.y + dy -s2);
  bb2.width = round(bb1.width*s);
  bb2.height = round(bb1.height*s);
}
bool TLD::trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &points1, vector<cv::Point2f> &points2){
  //TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
  //Forward-Backward tracking
  calcOpticalFlowPyrLK( img1,img2, points1, points2, status,similarity, window_size, level, term_criteria, lambda, 0);
  calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, FB_status,FB_error, window_size, level, term_criteria, lambda, 0);
  //Compute the real FB-error
  for( uint i= 0; i<points1.size(); ++i ){
        FB_error[i] = norm(pointsFB[i]-points1[i]);
  }
  //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
  normCrossCorrelation(img1,img2,points1,points2);
  return filterPts(points1,points2);
}

void TLD::normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
	Mat rec0(10,10,CV_8U);
	Mat rec1(10,10,CV_8U);
	Mat res(1,1,CV_32F);
	for (uint i = 0; i < points1.size(); i++) {
		if (status[i] == 1) {
			getRectSubPix( img1, Size(10,10), points1[i],rec0 );
			getRectSubPix( img2, Size(10,10), points2[i],rec1);
			matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);
			similarity[i] = ((float *)(res.data))[0];
		}
		else similarity[i] = 0.0;
	}
	rec0.release();
	rec1.release();
	res.release();
}
bool TLD::filterPts(vector<Point2f>& points1,vector<Point2f>& points2){
  //Get Error Medians
  float simmed = median(similarity);
  size_t i, k;
  for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])continue;
        if(similarity[i]>= simmed){
          points1[k] = points1[i];
          points2[k] = points2[i];
          FB_error[k] = FB_error[i];
          k++;
        }
    }
  if (k==0)return false;
  points1.resize(k);
  points2.resize(k);
  FB_error.resize(k);
  fbmed = median(FB_error);
  for( i=k = 0; i<points2.size(); ++i ){
      if( !status[i])continue;
      if(FB_error[i] <= fbmed){
        points1[k] = points1[i];
        points2[k] = points2[i];
        k++;
      }
  }
  points1.resize(k);
  points2.resize(k);
  if (k>0)return true;
  else return false;
}

