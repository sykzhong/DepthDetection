// FeatureMatch.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "FeatureMatch.h"

FeatureMatch::FeatureMatch()
{

}

FeatureMatch::~FeatureMatch()
{

}

void FeatureMatch::inputImage()
{
	vector<string> img_names;
	GlobalMethod::getFileNames("images", img_names);
	m_srcImages.resize(img_names.size());
	for (int i = 0; i < img_names.size(); i++)
	{
		m_srcImages[i] = imread(img_names[i]);		//syk:�ȶ�ȡ��ɫ�������ʱ�򻻳ɻ�ɫ��
	}
}

void FeatureMatch::extractFeature()
{
	m_keyPoints.clear();
	m_descriptors.clear();
	Ptr<Feature2D> surf = xfeatures2d::SURF::create(400, 4, 3, false, false);
	for (int i = 0; i < m_srcImages.size(); i++)
	{
		vector<KeyPoint> key_points;
		Mat descriptor;
		//ż�������ڴ����ʧ�ܵĴ���
		surf->detectAndCompute(m_srcImages[i], noArray(), key_points, descriptor);

		if (key_points.size() <= 10) 
			continue;
		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			colors[i] = m_srcImages[i].at<Vec3b>(p.y, p.x);
		}

		m_keyPoints.push_back(key_points);
		m_descriptors.push_back(descriptor);
		m_colors.push_back(colors);
	}
}

void FeatureMatch::matchFeatures()
{
	m_matches.clear();
	// n��ͼ������˳���� n-1 ��ƥ��
	// 1��2ƥ�䣬2��3ƥ�䣬3��4ƥ�䣬�Դ�����
	for (int i = 0; i < m_descriptors.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		matchFeatures(m_keyPoints[i], m_keyPoints[i + 1], m_descriptors[i], m_descriptors[i + 1], matches);
		m_matches.push_back(matches);
	}
}

void FeatureMatch::matchFeatures(vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches1;
	vector<vector<DMatch>> knn_matches2;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(descriptors1, descriptors2, knn_matches1, 2); //ͼ1ÿ����������ͼ2�������������Ƶ�
	matcher.knnMatch(descriptors2, descriptors1, knn_matches2, 2); //ͼ1ÿ����������ͼ2�������������Ƶ�

	int removed = ratioTest(knn_matches1);
	removed = ratioTest(knn_matches2);

	//�Գ��Լ��
	symmetryTest(knn_matches1, knn_matches2, matches);
}

int FeatureMatch::ratioTest(vector<vector<DMatch>>& matches)
{
	int removed = 0;

	// for all matches
	vector<vector<DMatch>>::iterator matchIterator;
	for (matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator) 
	{
		// if 2 NN has been identified
		if (matchIterator->size() > 1) {
			// check distance ratio  
			//ԭ��(*matchIterator)[0].distance ������ھ��룬(*matchIterator)[1].distance �Ǵν��ھ��룬����
			//(*matchIterator)[0].distance/(*matchIterator)[1].distance �϶�С��1
			//�������ںʹν��ڹ��ֽӽ�������ֵ�ӽ�1����ô�޷������Ǹ���������ƥ��㣬Ӧ���Ƴ��������
			//��������ratio����������ֵ�����������ʱ���Ƴ���ratio=0. 4������׼ȷ��Ҫ��ߵ�ƥ�䣻ratio = 0. 6������ƥ�����ĿҪ��Ƚ϶��ƥ�䣻 ratio = 0. 5��һ������¡�
			if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > 0.4) 
			{
				matchIterator->clear(); // remove match
				removed++;
			}
		}
		else 
		{ // does not have 2 neighbours
			matchIterator->clear(); // remove match
			removed++;
		}
	}
	return removed;
}

void FeatureMatch::symmetryTest(const vector<vector<DMatch>>& matches1, const vector<vector<DMatch>>& matches2, vector<DMatch>& symMatches)
{
	// for all matches image 1 -> image 2
	vector<vector<DMatch>>::const_iterator matchIterator1 = matches1.begin();
	vector<vector<DMatch>>::const_iterator matchIterator2 = matches2.begin();
	for (matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
	{
		if (matchIterator1->size() < 2) // ignore deleted matches 
			continue;

		// for all matches image 2 -> image 1
		for ( matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
		{

			if (matchIterator2->size() < 2) // ignore deleted matches 
				continue;

			// Match symmetry test �Գ��Բ���
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
				(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx)
			{
				// add symmetrical match  ��ӶԳƵ�ƥ��
				symMatches.push_back(DMatch((*matchIterator1)[0].queryIdx,
					(*matchIterator1)[0].trainIdx,
					(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}

void FeatureMatch::surfMatch()
{
	Ptr<SURF> surf;     
	surf = SURF::create(800);

	BFMatcher matcher;
	Mat  descriptorsA, descriptorsB;
	vector<DMatch> matches;

	surf->detectAndCompute(imgA, Mat(), keyA, descriptorsA);
	surf->detectAndCompute(imgB, Mat(), keyB, descriptorsB);

	m_pointA.resize(keyA.size());
	m_pointA.resize(keyB.size());
	KeyPoint::convert(keyA, m_pointA);
	KeyPoint::convert(keyA, m_pointB);

	matcher.match(descriptorsA, descriptorsB, matches);

	sort(matches.begin(), matches.end());  //ɸѡƥ���  
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
	drawMatches(imgA, keyA, imgB, keyB, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //����ƥ���  

	//std::vector<Point2f> obj;
	//std::vector<Point2f> scene;
	//std::vector<Point2f> obj_corners(4);
	//obj_corners[0] = Point(0, 0);
	//obj_corners[1] = Point(imgA.cols, 0);
	//obj_corners[2] = Point(imgA.cols, imgA.rows);
	//obj_corners[3] = Point(0, imgA.rows);
	//std::vector<Point2f> scene_corners(4);

	//Mat H = findHomography(obj, scene, RANSAC);      //Ѱ��ƥ���ͼ��  
	//perspectiveTransform(obj_corners, scene_corners, H);

	//line(outimg, scene_corners[0] + Point2f((float)imgA.cols, 0), scene_corners[1] + Point2f((float)imgA.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);       //����  
	//line(outimg, scene_corners[1] + Point2f((float)imgA.cols, 0), scene_corners[2] + Point2f((float)imgA.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	//line(outimg, scene_corners[2] + Point2f((float)imgA.cols, 0), scene_corners[3] + Point2f((float)imgA.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	//line(outimg, scene_corners[3] + Point2f((float)imgA.cols, 0), scene_corners[0] + Point2f((float)imgA.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
}

void FeatureMatch::RANSACMatch()
{
	int ptCount = (int)good_matches.size();
	if (ptCount < 100)
	{
		cout << "No enough match points." << endl;
		waitKey(0);
		return;
	}
	//��keypointת��ΪMat
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

	//��RANSAC���������������F
	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;
	findFundamentalMat(pA, pB, m_RANSACStatus, FM_RANSAC);

	//����Ұ�����
	int OutLinerCount = 0;
	for (int i = 0; i < ptCount; i++)
	{
		if (m_RANSACStatus[i] == 0)		//��ʾҰ��
		{
			OutLinerCount++;
		}
	}
	int InlinerCount = ptCount - OutLinerCount;
	cout << "�ڵ���Ϊ��" << InlinerCount << endl;

	//������ƥ��Ϊ�ڵ�������С
	m_InlierMatches.resize(InlinerCount);
	m_AInlier.resize(InlinerCount);
	m_BInlier.resize(InlinerCount);
	InlinerCount = 0;
	float inlier_minRx = imgA.cols;        //���ڴ洢�ڵ�����ͼ��С�����꣬�Ա�����ں�  

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
				inlier_minRx = m_BInlier[InlinerCount].x;   //�洢�ڵ�����ͼ��С������   
			InlinerCount++;
		}
	}
	//// ���ڵ�ת��ΪdrawMatches����ʹ�õĸ�ʽ   
	//vector<KeyPoint> key1_RANSAC(InlinerCount);
	//vector<KeyPoint> key2_RANSAC(InlinerCount);
	keyA_RANSAC.resize(InlinerCount);
	keyB_RANSAC.resize(InlinerCount);
	KeyPoint::convert(m_AInlier, keyA_RANSAC);
	KeyPoint::convert(m_BInlier, keyB_RANSAC);

	// ��ʾ����F������ڵ�ƥ��   
	Mat OutImage;
	drawMatches(imgA, keyA_RANSAC, imgB, keyB_RANSAC, m_InlierMatches, OutImage);
	namedWindow("RANSAC result2", CV_WINDOW_NORMAL);
	imshow("RANSAC result2", OutImage);
	waitKey(0);
}

bool FeatureMatch::findTransform()
{
	double focal_length_x = 1202.185, focal_length_y = 1202.363;
	double ux = 535.805, uy = 371.326;
	double focal_length = 0.5 * (focal_length_x + focal_length_y);
	Point2d principle_point(ux, uy);
	 
	return true;
}