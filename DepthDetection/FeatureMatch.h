#pragma once
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "GlobalMethod.h"
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class FeatureMatch
{
public:
	FeatureMatch();
	~FeatureMatch();
	void inputImage();
	void extractFeature();
	void matchFeatures();
	void matchFeatures(vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2, vector<DMatch>& matches);
	int ratioTest(vector<vector<DMatch>>& matches);
	void symmetryTest(const vector<vector<DMatch>>& matches1, const vector<vector<DMatch>>& matches2, vector<DMatch>& symMatches);

	void surfMatch();
	void RANSACMatch();
	bool findTransform();

private:
	vector<Mat> m_srcImages;
	vector<vector<KeyPoint>> m_keyPoints;
	vector<vector<Point2f>> m_Points;
	vector<Mat> m_descriptors;
	vector<vector<DMatch>> m_matches;
	vector<vector<Vec3b>> m_colors;

	Mat imgA;
	Mat imgB;
	vector<KeyPoint> keyA, keyB;
	vector<Point2f> m_pointA, m_pointB;
	vector<DMatch> good_matches;

	Mat outimg;

	Mat m_Fundamental;

	// �������������ڱ����ڵ��ƥ���ϵ   
	vector<Point2f> m_AInlier;
	vector<Point2f> m_BInlier;
	// ���ڵ�ת��ΪdrawMatches����ʹ�õĸ�ʽ   
	vector<KeyPoint> keyA_RANSAC;
	vector<KeyPoint> keyB_RANSAC;
	vector<DMatch> m_InlierMatches;

	Mat R, T;			//syk������ı任����
	Mat E;				//��������Essential Matrix)
};