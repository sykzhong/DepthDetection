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

	// �������������ڱ����ڵ��ƥ���ϵ   
	vector<Point2f> m_AInlier;
	vector<Point2f> m_BInlier;
	vector<DMatch> m_InlierMatches;

	// ���ڵ�ת��ΪdrawMatches����ʹ�õĸ�ʽ   
	vector<KeyPoint> keyA_RANSAC;
	vector<KeyPoint> keyB_RANSAC;
};