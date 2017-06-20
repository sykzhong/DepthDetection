#pragma once
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class DepthDetection
{
public:
	DepthDetection();
	~DepthDetection();
	void inputImage();
	void surfMatch();
	void RANSACMatch();

private:
	Mat imgA;
	Mat imgB;
	vector<KeyPoint> keyA, keyB;
	vector<DMatch> good_matches;

	Mat outimg;

	Mat m_Fundamental;

	// 这三个变量用于保存内点和匹配关系   
	vector<Point2f> m_AInlier;
	vector<Point2f> m_BInlier;
	vector<DMatch> m_InlierMatches;

	// 把内点转换为drawMatches可以使用的格式   
	vector<KeyPoint> keyA_RANSAC;
	vector<KeyPoint> keyB_RANSAC;
};