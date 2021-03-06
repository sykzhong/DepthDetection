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
	//二维特征点匹配相关
	void inputImage();
	void extractFeature();
	void matchFeatures();
	void matchFeatures(vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2, vector<DMatch>& matches);
	int ratioTest(vector<vector<DMatch>>& matches);
	void symmetryTest(const vector<vector<DMatch>>& matches1, const vector<vector<DMatch>>& matches2, vector<DMatch>& symMatches);
	//三维结构重建相关
	void initStructure();
	bool findTransform(vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask);		//根据两份匹配点，求解出两相机之间的相对R、T
	void getMatchPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch> matches, vector<Point2f>& out_p1, vector<Point2f>& out_p2);
	void getMatchColors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches, vector<Vec3b>& out_c1, vector<Vec3b>& out_c2);
	void maskOutPoints(vector<Point2f>& p1, Mat& mask);
	void maskOutColors(vector<Vec3b>& c1, Mat& mask);
	void reconstruct(Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure);
	void getObjpoints_Imgpoints(int match_index, int keypoint_index);			//match_index对应matches，keypoint_index对应keypoint（index均指图像编号）
	void savePoint2f();
	void savePoint3f();

	void completeStructure();
	void fusionStructure(int match_index, vector<Point3f> next_structure, vector<Vec3b> next_colors);
	void saveStructure(const string &filename);
private:
	vector<Mat> m_srcImages;
	Mat K;						//相机内参矩阵
	//Features相关
	vector<vector<KeyPoint>> m_keyPoints;
	vector<vector<Point2f>> m_Points;
	vector<Mat> m_descriptors;
	vector<vector<DMatch>> m_matches;
	vector<vector<Vec3b>> m_colors;

	//3d相关
	vector<Point3f> m_structure;
	vector<Vec3b> m_structure_color;
	vector<vector<int>> m_correspond_struct_idx;		//保存第i副图像中第j个特征点对应的structure中点的索引
	vector<Mat> m_rotations;
	vector<Mat> m_motions;
	vector<Point3f> m_objectPoints;
	vector<Point2f> m_imagePoints;
};