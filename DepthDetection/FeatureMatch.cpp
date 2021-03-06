// FeatureMatch.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "FeatureMatch.h"

FeatureMatch::FeatureMatch()
{
	float tmpK[3][3] = { 1202.185260913626, 0, 535.8053726984471,
	0, 1202.363115191773, 371.3262545972253,
	0, 0, 1 };
	K = Mat(3, 3, CV_32FC1, tmpK);
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
		m_srcImages[i] = imread(img_names[i]);		//syk:先读取彩色，处理的时候换成灰色？
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
		//偶尔出现内存分配失败的错误
		surf->detectAndCompute(m_srcImages[i], noArray(), key_points, descriptor);

		if (key_points.size() <= 10) 
			continue;

		vector<Point2f> points(key_points.size());
		KeyPoint::convert(key_points, points);		//将keypoints转换为points进行存储

		vector<Vec3b> colors(key_points.size());
		for (int j = 0; j < key_points.size(); ++j)
		{
			Point2f& p = key_points[j].pt;
			colors[j] = m_srcImages[i].at<Vec3b>(p.y, p.x);
		}

		m_keyPoints.push_back(key_points);
		m_Points.push_back(points);
		m_descriptors.push_back(descriptor);
		m_colors.push_back(colors);
	}
}

void FeatureMatch::matchFeatures()
{
	m_matches.clear();
	// n个图像，两两顺次有 n-1 对匹配
	// 1与2匹配，2与3匹配，3与4匹配，以此类推
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
	matcher.knnMatch(descriptors1, descriptors2, knn_matches1, 2); //图1每个特征点在图2中找两个最相似点
	matcher.knnMatch(descriptors2, descriptors1, knn_matches2, 2); //图1每个特征点在图2中找两个最相似点

	int removed = ratioTest(knn_matches1);
	removed = ratioTest(knn_matches2);

	//对称性检测
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
			//原理：(*matchIterator)[0].distance 是最近邻距离，(*matchIterator)[1].distance 是次近邻距离，所以
			//(*matchIterator)[0].distance/(*matchIterator)[1].distance 肯定小于1
			//如果最近邻和次近邻过分接近，即比值接近1，那么无法区分那个点才是最佳匹配点，应该移除这种情况
			//所以设置ratio比例，当比值大于这个比例时就移除。ratio=0. 4　对于准确度要求高的匹配；ratio = 0. 6　对于匹配点数目要求比较多的匹配； ratio = 0. 5　一般情况下。
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

			// Match symmetry test 对称性测试
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
				(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx)
			{
				// add symmetrical match  添加对称的匹配
				symMatches.push_back(DMatch((*matchIterator1)[0].queryIdx,
					(*matchIterator1)[0].trainIdx,
					(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}

/***********以下为三维重建相关程序**************/
//
//
//
void FeatureMatch::getMatchPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch> matches, vector<Point2f>& out_p1, vector<Point2f>& out_p2)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void FeatureMatch::getMatchColors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches, vector<Vec3b>& out_c1, vector<Vec3b>& out_c2)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}


bool FeatureMatch::findTransform(vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	//根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length_x = K.at<float>(0), focal_length_y = K.at<float>(4);
	double ux = K.at<float>(2), uy = K.at<float>(5);
	double focal_length = 0.5 * (focal_length_x + focal_length_y);
	Point2d principle_point(ux, uy);

	//根据匹配点求取本征矩阵E，使用RANSAC，进一步排除失配点,mask是输出矩阵1068行（keypoints个数）1列，E是3行3列
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty())
		return false;

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size() << endl;
	//对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
		return false;

	//分解本征矩阵E，获取相对变换，R，T都是3行3列
	//得到本征矩阵后，再使用另一个函数recoverPose对本征矩阵进行分解，并返回两相机之间的相对变换R和T。
	//注意这里的T是在第二个相机的坐标系下表示的，也就是说，其方向从第二个相机指向第一个相机（即世界坐标系所在的相机），且它的长度等于1。
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.7)
		return false;

	return true;
}

void FeatureMatch::maskOutPoints(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void FeatureMatch::maskOutColors(vector<Vec3b> &c1, Mat &mask)
{
	vector<Vec3b> c1_copy = c1;
	c1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			c1.push_back(c1_copy[i]);
	}
}

void FeatureMatch::reconstruct(Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure)
{
	//两个相机的投影矩阵[R T]，triangulatePoints只支持float型
	Mat proj1(3, 4, CV_32FC1); //3行4列
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK*proj1;
	proj2 = fK*proj2;

	//三角重建
	Mat s; // 4xN 在齐次坐标系之中重构的向量
	triangulatePoints(proj1, proj2, p1, p2, s);

	structure.clear();
	structure.reserve(s.cols); //vector通过 reserve() 来申请特定大小的时候总是按指数边界来增大其内部缓冲区
	for (int i = 0; i < s.cols; ++i)
	{
		Mat_<float> col = s.col(i);  //col也是个数组
		col /= col(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
		structure.push_back(Point3f(col(0), col(1), col(2)));
	}

	//structure 中元素的个数即p的个数，存储的是point3f，即点的坐标
}

void FeatureMatch::getObjpoints_Imgpoints(int match_index, int keypoint_index)
{
	m_objectPoints.clear();
	m_imagePoints.clear();
	vector<DMatch> matches = m_matches[match_index];
	for (int i = 0; i < matches.size(); i++)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = m_correspond_struct_idx[match_index][query_idx];
		if (struct_idx < 0)
			continue;

		m_objectPoints.push_back(m_structure[struct_idx]);
		m_imagePoints.push_back(m_keyPoints[keypoint_index][query_idx].pt);
	}
}

void FeatureMatch::initStructure()
{
	vector<Point2f> p1, p2;
	//vector<Vec3b> colors;		//syk:用于存储三维图的彩色像素点像素？
	vector<Vec3b> c2;
	Mat R, T;	//旋转矩阵和平移向量（由相机二至相机一）
	Mat mask;	//mask中大于零的点代表匹配点，等于零代表失配点
	getMatchPoints(m_keyPoints[0], m_keyPoints[1], m_matches[0], p1, p2);
	getMatchColors(m_colors[0], m_colors[1], m_matches[0], m_structure_color, c2);

	findTransform(p1, p2, R, T, mask);

	maskOutPoints(p1, mask);
	maskOutPoints(p2, mask);
	maskOutColors(m_structure_color, mask);

	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);

	//对头两幅图像进行三维重建，输出结果：vector<Point3f>& structure
	// 第一幅相机坐标系作为世界坐标系，所以它的矩阵为R0，T0

	reconstruct(R0, T0, R, T, p1, p2, m_structure);

	//保存变换矩阵
	m_rotations = { R0, R };
	m_motions = { T0, T };

	m_correspond_struct_idx.clear();
	m_correspond_struct_idx.resize(m_keyPoints.size());
	for (int i = 0; i < m_keyPoints.size(); i++)
	{
		m_correspond_struct_idx[i].resize(m_keyPoints[i].size(), -1);
	}

	//填写头两幅图像的结构索引correspond_struct_idx
	int idx = 0;
	vector<DMatch>& matches = m_matches[0];

	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
			continue;
		m_correspond_struct_idx[0][matches[i].queryIdx] = idx;
		m_correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}
	getObjpoints_Imgpoints(0, 0);
	savePoint2f();
	savePoint3f();
}

void FeatureMatch::savePoint2f()
{
	ofstream outfile(".\\Viewer\\point2d.txt", ios::out);
	if (!outfile.is_open())
	{
		cout << " the file open fail" << endl;
		exit(1);
	}
	for (int i = 0; i<m_imagePoints.size(); i++)
	{

		outfile << m_imagePoints[i].x << " ";
		outfile << m_imagePoints[i].y;
		outfile << "\n";
	}
	outfile.close();
}

void FeatureMatch::savePoint3f()
{

	ofstream outfile(".\\Viewer\\point3d.txt", ios::out);


	if (!outfile.is_open())
	{
		cout << " the file open fail" << endl;
		exit(1);
	}
	for (int i = 0; i<m_objectPoints.size(); i++)
	{
		outfile << m_objectPoints[i].x << " ";
		outfile << m_objectPoints[i].y << " ";
		outfile << m_objectPoints[i].z;
		outfile << "\n";
	}
	outfile.close();
}

void FeatureMatch::completeStructure()
{
	for (int i = 1; i < m_matches.size(); i++)
	{
		Mat r, R, T;		//r存储i与0之间的旋转关系，即与第一个相机（世界坐标）之间的关系
		getObjpoints_Imgpoints(i, i + 1);
		//求解变换矩阵
		solvePnPRansac(m_objectPoints, m_imagePoints, K, noArray(), r, T);
		//将旋转向量转换为旋转矩阵
		Rodrigues(r, R);
		//保存变换矩阵
		m_rotations.push_back(R);
		m_motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		getMatchPoints(m_keyPoints[i], m_keyPoints[i + 1], m_matches[i], p1, p2);
		getMatchColors(m_colors[i], m_colors[i + 1], m_matches[i], c1, c2);

		vector<Point3f> next_structure;
		reconstruct(m_rotations[i], m_motions[i], R, T, p1, p2, next_structure);
		fusionStructure(i, next_structure, c1);
	}
}

void FeatureMatch::fusionStructure(int match_index, vector<Point3f> next_structure, vector<Vec3b> next_colors)
{
	vector<DMatch> matches = m_matches[match_index];
	for (int i = 0; i < matches.size(); i++)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		vector<int> &struct_indices = m_correspond_struct_idx[i];
		vector<int> &next_struct_indices = m_correspond_struct_idx[i + 1];

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0)		//若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		//若该点在空间中未存在，将该点加入到结构中，且这对匹配点的空间点索引都为新加入的点的索引
		m_structure.push_back(next_structure[i]);
		m_structure_color.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = m_structure.size() - 1;
	}
}

void FeatureMatch::saveStructure(const string &filename)
{
	int n = (int)m_rotations.size();

	FileStorage fs(filename, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)m_structure.size();

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << m_rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << m_motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < m_structure.size(); ++i)
	{
		fs << m_structure[i];
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < m_structure_color.size(); ++i)
	{
		fs << m_structure_color[i];
	}
	fs << "]";

	fs.release();
}