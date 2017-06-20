// DepthDetection.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "DepthDetection.h"

DepthDetection::DepthDetection()
{

}

DepthDetection::~DepthDetection()
{

}

void DepthDetection::inputImage()
{
	imgA = imread("Surf1.JPG", IMREAD_GRAYSCALE);    //读取灰度图像  
	imgB = imread("Surf2.JPG", IMREAD_GRAYSCALE);
}

void DepthDetection::surfMatch()
{
	Ptr<SURF> surf;     
	surf = SURF::create(800);

	BFMatcher matcher;
	Mat  descriptorsA, descriptorsB;
	vector<DMatch> matches;

	surf->detectAndCompute(imgA, Mat(), keyA, descriptorsA);
	surf->detectAndCompute(imgB, Mat(), keyB, descriptorsB);

	matcher.match(descriptorsA, descriptorsB, matches);

	sort(matches.begin(), matches.end());  //筛选匹配点  
	float max_dist = matches.back().distance, min_dist = matches.front().distance;
	for (int i = 0; i < matches.size(); i++)
	{
		//if (matches[i].distance < 0.3*max_dist)
		if (matches[i].distance < 0.6*max_dist)
			good_matches.push_back(matches[i]);
		//if (good_matches.size() >= 200)
		//	break;
	}
	Mat outimg;
	drawMatches(imgA, keyA, imgB, keyB, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  

	//std::vector<Point2f> obj;
	//std::vector<Point2f> scene;
	//std::vector<Point2f> obj_corners(4);
	//obj_corners[0] = Point(0, 0);
	//obj_corners[1] = Point(imgA.cols, 0);
	//obj_corners[2] = Point(imgA.cols, imgA.rows);
	//obj_corners[3] = Point(0, imgA.rows);
	//std::vector<Point2f> scene_corners(4);

	//Mat H = findHomography(obj, scene, RANSAC);      //寻找匹配的图像  
	//perspectiveTransform(obj_corners, scene_corners, H);

	//line(outimg, scene_corners[0] + Point2f((float)imgA.cols, 0), scene_corners[1] + Point2f((float)imgA.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);       //绘制  
	//line(outimg, scene_corners[1] + Point2f((float)imgA.cols, 0), scene_corners[2] + Point2f((float)imgA.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	//line(outimg, scene_corners[2] + Point2f((float)imgA.cols, 0), scene_corners[3] + Point2f((float)imgA.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	//line(outimg, scene_corners[3] + Point2f((float)imgA.cols, 0), scene_corners[0] + Point2f((float)imgA.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
}

void DepthDetection::RANSACMatch()
{
	int ptCount = (int)good_matches.size();
	if (ptCount < 100)
	{
		cout << "No enough match points." << endl;
		waitKey(0);
		return;
	}
	//将keypoint转换为Mat
	Point2f pt;
	Mat pA(ptCount, 2, CV_32F);
	Mat pB(ptCount, 2, CV_32F);
	for (int i = 0; i < ptCount; i++)
	{
		pt = keyA[good_matches[i].queryIdx].pt;
		pA.at<float>(i, 0) = pt.x;
		pA.at<float>(i, 1) = pt.y;

		pt = keyB[good_matches[i].trainIdx].pt;
		pB.at<float>(i, 0) = pt.x;
		pB.at<float>(i, 1) = pt.y;
	}

	//用RANSAC方法计算基础矩阵F
	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;
	findFundamentalMat(pA, pB, m_RANSACStatus, FM_RANSAC);

	//计算野点个数
	int OutLinerCount = 0;
	for (int i = 0; i < ptCount; i++)
	{
		if (m_RANSACStatus[i] == 0)		//表示野点
		{
			OutLinerCount++;
		}
	}
	int InlinerCount = ptCount - OutLinerCount;
	cout << "内点数为：" << InlinerCount << endl;

	//将变量匹配为内点数量大小
	m_InlierMatches.resize(InlinerCount);
	m_AInlier.resize(InlinerCount);
	m_BInlier.resize(InlinerCount);
	InlinerCount = 0;
	float inlier_minRx = imgA.cols;        //用于存储内点中右图最小横坐标，以便后续融合  

	for (int i = 0; i < ptCount; i++)
	{
		if (m_RANSACStatus[i] != 0)
		{
			m_AInlier[InlinerCount].x = pA.at<float>(i, 0);
			m_AInlier[InlinerCount].y = pA.at<float>(i, 1);
			m_BInlier[InlinerCount].x = pB.at<float>(i, 0);
			m_BInlier[InlinerCount].y = pB.at<float>(i, 1);
			m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
			m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
			if (m_BInlier[InlinerCount].x<inlier_minRx)
				inlier_minRx = m_BInlier[InlinerCount].x;   //存储内点中右图最小横坐标   
			InlinerCount++;
		}
	}
	//// 把内点转换为drawMatches可以使用的格式   
	//vector<KeyPoint> key1_RANSAC(InlinerCount);
	//vector<KeyPoint> key2_RANSAC(InlinerCount);
	keyA_RANSAC.resize(InlinerCount);
	keyB_RANSAC.resize(InlinerCount);
	KeyPoint::convert(m_AInlier, keyA_RANSAC);
	KeyPoint::convert(m_BInlier, keyB_RANSAC);

	// 显示计算F过后的内点匹配   
	Mat OutImage;
	drawMatches(imgA, keyA_RANSAC, imgB, keyB_RANSAC, m_InlierMatches, OutImage);
	namedWindow("RANSAC result2", CV_WINDOW_NORMAL);
	imshow("RANSAC result2", OutImage);
	waitKey(0);
}