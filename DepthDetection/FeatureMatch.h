#pragma once
#include <iostream>
#include <fstream>
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

	void initStructure();
	bool findTransform(vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask);		//��������ƥ��㣬���������֮������R��T
	void getMatchPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch> matches, vector<Point2f>& out_p1, vector<Point2f>& out_p2);
	void getMatchColors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches, vector<Vec3b>& out_c1, vector<Vec3b>& out_c2);
	void maskOutPoints(vector<Point2f>& p1, Mat& mask);
	void maskOutColors(vector<Vec3b>& c1, Mat& mask);
	void reconstruct(Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure);
	void getObjpoints_Imgpoints();
	void savePoint2f();
	void savePoint3f();

private:
	vector<Mat> m_srcImages;
	Mat K;						//����ڲξ���
	//Features���
	vector<vector<KeyPoint>> m_keyPoints;
	vector<vector<Point2f>> m_Points;
	vector<Mat> m_descriptors;
	vector<vector<DMatch>> m_matches;
	vector<vector<Vec3b>> m_colors;

	//3d���
	vector<Point3f> m_structure;
	vector<vector<int>> m_correspond_struct_idx;		//�����i��ͼ���е�j���������Ӧ��structure�е������
	vector<Mat> m_rotations;
	vector<Mat> m_motions;
	vector<Point3f> m_objectPoints;
	vector<Point2f> m_imagePoints;

	/************************/
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