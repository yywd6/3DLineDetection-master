#include "LineDetection3D.h"
#include <omp.h>
#include <unordered_set>
#include "CommonFunctions.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

LineDetection3D::LineDetection3D()
{
}

LineDetection3D::~LineDetection3D()
{
}

void LineDetection3D::run( PointCloud<double> &data, PointCloud<double>& projectData, int k, std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines, std::vector<double> &ts  )
{
	this->pointData = data;
	this->projectPointData = projectData;
	this->pointNum = data.pts.size();
	this->k = k;

	// step1: point cloud segmentation
	double totalTime = 0.0;
	CTimer timer;
	char msg[1024];

	timer.Start();
	cout<<endl<<endl;
	cout<<"Step1: Point Cloud Segmentation ..."<<endl;
	std::vector<std::vector<int> > regions;
	pointCloudSegmentation( regions );
	timer.Stop();

	


	totalTime += timer.GetElapsedSeconds();
	timer.PrintElapsedTimeMsg(msg);
	printf("  Point Cloud Segmentation Time: %s.\n\n", msg);
	ts.push_back(timer.GetElapsedSeconds());

	cout << "Step1: Segmentation has finshed ..." << endl;

	return;

	// step2: plane based 3D line detection
	timer.Start();
	cout<<"Step2: Plane Based 3D LineDetection ..."<<endl;
	planeBased3DLineDetection( regions, planes );
	timer.Stop();
	totalTime += timer.GetElapsedSeconds();
	timer.PrintElapsedTimeMsg(msg);
	printf("  Plane Based 3D LineDetection Time: %s.\n\n", msg);
	ts.push_back(timer.GetElapsedSeconds());

	// step3: post processing
	timer.Start();
	cout<<"Step3: Post Processing ..."<<endl;
	postProcessing( planes, lines );
	timer.Stop();
	totalTime += timer.GetElapsedSeconds();
	timer.PrintElapsedTimeMsg(msg);
	printf("  Post Processing Time: %s.\n\n", msg);
	ts.push_back(timer.GetElapsedSeconds());

	printf("Total Time: %lf.\n\n", totalTime);
}


void writeOutPlane(string filePath, std::vector<std::vector<int> > regions)
{
	
}

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

void LineDetection3D::pointCloudSegmentation( std::vector<std::vector<int> > &regions )
{
	cout<<"----- Normal Calculation ..."<<endl;
	PCAFunctions pcaer;
	pcaer.Ori_PCA( this->pointData, this->k, this->pcaInfos, this->scale, this->magnitd );
	
	cout<<"----- Region Growing ..."<<endl;
	double thAngle = 15.0/180.0*CV_PI; //原本是15
	regionGrow( thAngle, regions );

	// step3: region merging
	cout<<"----- Region Merging ..."<<endl;
	double thAnglePatch = thAngle;
	regionMerging( thAnglePatch, regions );


	cout << "----- Plane Merging ..." << endl;
	planeMerging(regions, thAngle);

	// write out bounding polygon result
	string fileEdgePoints = "C:\\Users\\89183\\Desktop\\WVData\\partResults\\part4Result.txt";
	FILE* fp2 = fopen(fileEdgePoints.c_str(), "w");
	for (int p = 0; p < regions.size(); ++p)
	{
		std::random_device rd;

		// 结合时间和 random_device 的值来生成一个更复杂的种子
		unsigned seed = rd() ^ std::chrono::steady_clock::now().time_since_epoch().count() * 80;

		// 使用生成的种子初始化 mt19937 伪随机数生成器
		std::mt19937 gen(seed);

		// 定义一个均匀分布的范围为 0 到 255
		std::uniform_int_distribution<> dis(0, 255);

		// 生成三个随机的8位无符号整数，分别赋值给 R, G, B
		uint8_t R = dis(gen)+ 50;
		uint8_t G = dis(gen) + 50;
		uint8_t B = dis(gen) + 50;

		for (int i = 0; i < regions[p].size(); ++i)
		{
			fprintf(fp2, "%.6lf   %.6lf   %.6lf    ", pointData.pts[regions[p][i]].x, pointData.pts[regions[p][i]].y, pointData.pts[regions[p][i]].z);
			fprintf(fp2, "%d   %d   %d\n", R, G, B);
		}

	}
	fclose(fp2);

}


void LineDetection3D::regionGrow( double thAngle, std::vector<std::vector<int> > &regions )
{
	double thNormal = cos(thAngle);  

	// sort according to the curvature of points
	std::vector<std::pair<int,double> > idxSorted( this->pointNum ); //创建一个容器对，第一个是int（点的数量），第二个是double（曲率大小），容器数量为this->pointNum
	for ( int i=0; i<this->pointNum; ++i )
	{
		idxSorted[i].first = i;
		idxSorted[i].second = pcaInfos[i].lambda0;
	}
	std::sort( idxSorted.begin(), idxSorted.end(), [](const std::pair<int,double>& lhs, const std::pair<int,double>& rhs) { return lhs.second < rhs.second; } );

	// get the initial clusters
	double percent = 0.9;
	int idx = int(this->pointNum*percent);
	std::vector<int> isUsed( this->pointNum, 0 );
	for ( int i=0; i<idx; ++i )
	{
		int idxStrater = idxSorted[i].first;
		if ( isUsed[idxStrater] ) { continue; }
		cv::Matx31d normalStarter = pcaInfos[idxStrater].normal;
		double xStrater = pointData.pts[idxStrater].x, yStrater = pointData.pts[idxStrater].y, zStrater = pointData.pts[idxStrater].z;
		double thRadius2 = pow(50*pcaInfos[idxStrater].scale, 2);

		std::vector<int> clusterTemp;
		clusterTemp.reserve(10000);
		clusterTemp.push_back( idxStrater );
		int count = 0;
		while( count < clusterTemp.size() )
		{
			int idxSeed = clusterTemp[count];
			cv::Matx31d normalSeed = pcaInfos[idxSeed].normal;
			double thOrtho = pcaInfos[idxSeed].scale;

			// point cloud collection
			int num = pcaInfos[idxSeed].idxAll.size();
			for( int j = 0; j < num; ++j )
			{
				int idxCur = pcaInfos[idxSeed].idxAll[j];
				if (isUsed[idxCur])
				{
					continue;
				}

				// judgement1: normal deviation
				cv::Matx31d normalCur = pcaInfos[idxCur].normal;
				double normalDev = abs(normalCur.val[0] * normalStarter.val[0] + normalCur.val[1] * normalStarter.val[1] + normalCur.val[2] * normalStarter.val[2]);
				//double normalDev = abs(normalCur.val[0] * normalSeed.val[0] + normalCur.val[1] * normalSeed.val[1] + normalCur.val[2] * normalSeed.val[2]);
				if (normalDev < thNormal)
				{
					continue;
				}

				// judgement2: orthogonal distance
				double dx = pointData.pts[idxCur].x - xStrater;
				double dy = pointData.pts[idxCur].y - yStrater;
				double dz = pointData.pts[idxCur].z - zStrater;
				double dOrtho = abs(dx * normalCur.val[0] + dy * normalCur.val[1] + dz * normalCur.val[2]);
				if (dOrtho > thOrtho)
				{
					continue;
				}

				// judgement3: parallel distance
				double dPara = dx*dx + dy*dy + dz*dz;
				if (dPara > thRadius2)
				{
					continue;
				}

				clusterTemp.push_back( idxCur );
				isUsed[idxCur] = 1;
			}
			count ++;
		}

		if ( clusterTemp.size() > 30) //raw code :  clusterTemp.size() > 30 
		{
			regions.push_back( clusterTemp );
		}
		else
		{
			for (int j=0; j<clusterTemp.size(); ++j)
			{
				isUsed[clusterTemp[j]] = 0;
			}
		}
	}
}

void LineDetection3D::regionMerging( double thAngle, std::vector<std::vector<int> > &regions )
{
	double thRegionSize = 600000;

	// step1: plane fitting via PCA for each region
	std::vector<PCAInfo> patches;
	patches.resize( regions.size() );

#pragma omp parallel for
	for ( int i=0; i<regions.size(); ++i )
	{
		int pointNumCur = regions[i].size();
		std::vector<std::vector<double> > pointDataCur(pointNumCur);
		for ( int j=0; j<pointNumCur; ++j )
		{
			pointDataCur[j].resize(3);
			pointDataCur[j][0] = this->pointData.pts[regions[i][j]].x;
			pointDataCur[j][1] = this->pointData.pts[regions[i][j]].y;
			pointDataCur[j][2] = this->pointData.pts[regions[i][j]].z;
		}

		PCAFunctions pcaer;
		pcaer.PCASingle( pointDataCur, patches[i] );

		patches[i].idxAll = regions[i];
		double scaleAvg = 0.0;
		for ( int j=0; j<patches[i].idxIn.size(); ++j )
		{
			int idx = regions[i][patches[i].idxIn[j]];
			patches[i].idxIn[j] = idx;
			scaleAvg += pcaInfos[idx].scale;
		}
		scaleAvg /= patches[i].idxIn.size();
		patches[i].scale = 5.0 * scaleAvg;			//原来是5.0 * scaleAvg
	}

	// get the patch label of each point
	std::vector<int> label( this->pointNum, -1 );
#pragma omp parallel for
	for ( int i=0; i<regions.size(); ++i )
	{
		for ( int j=0; j<regions[i].size(); ++j )
		{
			int id = regions[i][j];
			label[id] = i;
		}
	}

	// step2: find adjacent patches
	std::vector<std::vector<int> > patchAdjacent( patches.size() );
#pragma omp parallel for
	for ( int i=0; i<patches.size(); ++i )
	{
		std::vector<int> patchAdjacentTemp;
		std::vector<std::vector<int> > pointAdjacentTemp;
		for ( int j=0; j<patches[i].idxIn.size(); ++j )
		{
			int id = patches[i].idxIn[j];
			for ( int m=0; m<pcaInfos[id].idxIn.size(); ++m )
			{
				int idPoint = pcaInfos[id].idxIn[m];
				int labelPatch = label[idPoint];
				if ( labelPatch == i || labelPatch < 0 )
				{
					continue;
				}

				bool isNeighbor = false;
				for ( int n=0; n<pcaInfos[idPoint].idxIn.size(); ++n )
				{
					if ( pcaInfos[idPoint].idxIn[n] == id )
					{
						isNeighbor = true;
					}
				}
				if ( ! isNeighbor )
				{
					continue;
				}

				// accept the patch as a neighbor
				bool isIn = false;
				int n = 0;
				for ( n=0; n<patchAdjacentTemp.size(); ++n )
				{
					if ( patchAdjacentTemp[n] == labelPatch )
					{
						isIn = true;
						break;
					}
				}

				if ( isIn )
				{
					pointAdjacentTemp[n].push_back( idPoint );
				}
				else
				{
					patchAdjacentTemp.push_back( labelPatch );

					std::vector<int> temp;
					temp.push_back( idPoint );
					pointAdjacentTemp.push_back( temp );
				}
			}
		}

		// repetition removal
		for ( int j=0; j<pointAdjacentTemp.size(); ++j )
		{
			std::sort(pointAdjacentTemp[j].begin(), pointAdjacentTemp[j].end());  
			vector<int>::iterator new_end = unique(pointAdjacentTemp[j].begin(), pointAdjacentTemp[j].end());
			pointAdjacentTemp[j].erase(new_end, pointAdjacentTemp[j].end());

			if ( pointAdjacentTemp[j].size() >= 3 )
			{
				patchAdjacent[i].push_back( patchAdjacentTemp[j] );
			}
		}
	}

	// try to merge adjacent patch
	regions.clear();
	std::vector<int> mergedIndex( patches.size(), 0 );
	for ( int i=0; i<patches.size(); ++i )
	{
		if ( !mergedIndex[i] )
		{
			int idxStarter = i;
			cv::Matx31d normalStarter = patches[idxStarter].normal;
			cv::Matx31d ptStarter = patches[idxStarter].planePt;

			std::vector<int> patchIdx;
			patchIdx.push_back( idxStarter );

			int count = 0;
			int totalPoints = 0;
			bool isEnough = false;
			while ( count < patchIdx.size() )
			{
				int idxSeed = patchIdx[count];
				cv::Matx31d normalSeed = patches[idxSeed].normal;
				cv::Matx31d ptSeed = patches[idxSeed].planePt;
				double thOrtho = patches[idxSeed].scale; //原本是patches[idxSeed].scale

				for ( int j=0; j<patchAdjacent[idxSeed].size(); ++j )
				{
					int idxCur = patchAdjacent[idxSeed][j];

					if ( mergedIndex[idxCur] )
					{
						continue;
					}

					cv::Matx31d normalCur = patches[idxCur].normal;
					cv::Matx31d ptCur = patches[idxCur].planePt;

					// plane angle deviation and distance
					double devAngle = 0.0;
					double devDis = 0.0;
					double thDev = 0.0;

					cv::Matx31d ptVector1 = ptCur - ptStarter;
					cv::Matx31d ptVector2 = ptCur - ptSeed;
					devAngle = acos( normalStarter.val[0] * normalCur.val[0] + normalStarter.val[1] * normalCur.val[1] + normalStarter.val[2] * normalCur.val[2] );
					//devDis = abs( normalSeed.val[0] * ptVector2.val[0] + normalSeed.val[1] * ptVector2.val[1] + normalSeed.val[2] * ptVector2.val[2] );
					devDis = abs( normalStarter.val[0] * ptVector1.val[0] + normalStarter.val[1] * ptVector1.val[1] + normalStarter.val[2] * ptVector1.val[2] );

					if (min(devAngle, fabs(CV_PI - devAngle)) < thAngle && devDis < thOrtho) //( min( devAngle, fabs( CV_PI - devAngle ) ) < thAngle && devDis < thOrtho ) °
					{
						patchIdx.push_back( idxCur );
						mergedIndex[idxCur] = 1;

						totalPoints += patches[idxCur].idxAll.size();
						if (totalPoints > thRegionSize) //raw code：totalPoints > thRegionSize 
						{
							isEnough = true;
							break;
						}
					}
				}

				if ( isEnough )
				{
					break;
				}
				count ++;
			}

			// create a new cluster
			std::vector<int> patchNewCur;
			/*for ( int j=0; j<patchIdx.size(); ++j )
			{
				int idx = patchIdx[j];

				for ( int m=0; m<patches[idx].idxAll.size(); ++m )
				{
					patchNewCur.push_back( patches[idx].idxAll[m] );
				}
			}*/

			std::unordered_set<int> uniquePoints;  // 使用 unordered_set 进行去重

			for (int j = 0; j < patchIdx.size(); ++j)
			{
				int idx = patchIdx[j];

				for (int m = 0; m < patches[idx].idxAll.size(); ++m)
				{
					int point = patches[idx].idxAll[m];
					if (uniquePoints.find(point) == uniquePoints.end())  // 如果点没有在集合中
					{
						patchNewCur.push_back(point);   // 加入到 patchNewCur
						uniquePoints.insert(point);     // 插入到去重集合
					}
				}
			}

			// 
			if (patchNewCur.size() > 500)	//raw code : patchNewCur.size() > 100
			{
				regions.push_back( patchNewCur );
			}
		}
	}
}

void LineDetection3D::planeMerging(std::vector<std::vector<int> >& regions, double thAngle )
{
	std::sort(regions.begin(), regions.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
		return a.size() > b.size();
		});		//Sort by the number points of each region, form large to small

	//for (int p = 0; p < regions.size(); ++p)
	//{
	//	// 生成文件名，例如 "plane0.txt", "plane1.txt" 等
	//	std::string filename = "c:\\users\\89183\\desktop\\wvdata\\eachplane\\plane" + std::to_string(p) + ".txt";
	//	FILE* fp = fopen(filename.c_str(), "w");  // 为每个区域创建一个独立文件

	//	if (fp == nullptr) {
	//		std::cerr << "error opening file: " << filename << std::endl;
	//		continue;
	//	}

	//	std::random_device rd;
	//	unsigned seed = rd() ^ std::chrono::steady_clock::now().time_since_epoch().count() * 80;
	//	std::mt19937 gen(seed);
	//	std::uniform_int_distribution<> dis(0, 255);
	//	uint8_t r = dis(gen) + 50;
	//	uint8_t g = dis(gen) + 50;
	//	uint8_t b = dis(gen) + 50;

	//	for (int i = 0; i < regions[p].size(); ++i)
	//	{
	//		// 将点数据写入每个单独的文件中
	//		fprintf(fp, "%.6lf   %.6lf   %.6lf    ", pointData.pts[regions[p][i]].x, pointData.pts[regions[p][i]].y, pointData.pts[regions[p][i]].z);
	//		fprintf(fp, "%d   %d   %d\n", r, g, b);
	//	}

	//	fclose(fp);  // 关闭当前文件
	//}


	// step1: plane fitting via PCA for each region
	std::vector<PCAInfo> patches;
	patches.resize(regions.size());

#pragma omp parallel for
	for (int i = 0; i < regions.size(); ++i)
	{
		int pointNumCur = regions[i].size();
		std::vector<std::vector<double> > pointDataCur(pointNumCur);
		for (int j = 0; j < pointNumCur; ++j)
		{
			pointDataCur[j].resize(3);
			pointDataCur[j][0] = this->pointData.pts[regions[i][j]].x;
			pointDataCur[j][1] = this->pointData.pts[regions[i][j]].y;
			pointDataCur[j][2] = this->pointData.pts[regions[i][j]].z;
		}

		PCAFunctions pcaer;
		pcaer.PCASingle(pointDataCur, patches[i]);

		patches[i].idxAll = regions[i];
		double scaleAvg = 0.0;
		for (int j = 0; j < patches[i].idxIn.size(); ++j)
		{
			int idx = regions[i][patches[i].idxIn[j]];
			patches[i].idxIn[j] = idx;
			scaleAvg += pcaInfos[idx].scale;
		}
		scaleAvg /= patches[i].idxIn.size();
		patches[i].scale = 5.0 * scaleAvg;			//原来是5.0 * scaleAvg
	}
	
	//start merging
	for (size_t baseIdx = 0; baseIdx < regions.size(); ++baseIdx)
	{
		if (regions[baseIdx].empty()) continue;
		std::vector<int> parallelPlane;

		for (size_t i = baseIdx + 1; i < regions.size(); ++i)
		{
			// 使用向量点积计算夹角
			double dotProduct = patches[i].normal.dot(patches[baseIdx].normal);
			// 修正浮点误差，使点积范围在 [-1, 1] 之间
			dotProduct = std::max(-1.0, std::min(1.0, dotProduct));
			double devAngle = acos(dotProduct);

			if (min(devAngle, fabs(CV_PI - devAngle)) < thAngle)
			{
				parallelPlane.push_back(i);
			}
		}

		int i = 0;
		while (i < parallelPlane.size())  // 使用 while 循环替代 for 循环
		{
			auto curPlane = parallelPlane[i];
			if (regions[curPlane].empty())
			{
				i++;
				continue;
			}
			// 计算方向向量
			cv::Matx31d direction = patches[baseIdx].middlePoints - patches[curPlane].middlePoints;

			// 使用点积计算 devAngle
			double dotProduct = direction.dot(patches[baseIdx].normal);
			// 修正浮点误差，使点积范围在 [-1, 1] 之间
			dotProduct = std::max(-1.0, std::min(1.0, dotProduct));
			double devAngle2 = acos(dotProduct);

			double minAngle = CV_PI * 85.0 / 180.0; // 75°
			double maxAngle = CV_PI * 95.0 / 180.0; // 105°

			if (minAngle < std::min(devAngle2, fabs(CV_PI - devAngle2)) && std::min(devAngle2, fabs(CV_PI - devAngle2)) < maxAngle)
			{
				float baseDistanse = std::sqrt(pow((patches[baseIdx].boundMax.val[0] - patches[baseIdx].boundMin.val[0]), 2.0) +
					pow((patches[baseIdx].boundMax.val[1] - patches[baseIdx].boundMin.val[1]), 2.0));
				float curDistanse = std::sqrt(pow((patches[curPlane].boundMax.val[0] - patches[curPlane].boundMin.val[0]), 2.0) +
					pow((patches[curPlane].boundMax.val[1] - patches[curPlane].boundMin.val[1]), 2.0));
				float pointDistance = std::sqrt(pow((patches[baseIdx].middlePoints.val[0] - patches[curPlane].middlePoints.val[0]), 2.0) +
					pow((patches[baseIdx].middlePoints.val[1] - patches[curPlane].middlePoints.val[1]), 2.0));

				float thDis = pointDistance - (baseDistanse + curDistanse) / 2;

				if (thDis < 0.1)
				{
					// 将 regions[i] 中的点加入 regions[baseIdx]
					regions[baseIdx].insert(regions[baseIdx].end(), regions[curPlane].begin(), regions[curPlane].end());
					// 删除 regions[i] 的点
					regions[curPlane].clear(); // 不直接删除，只清空，以保持索引一致性

					//更新信息
					int pointNumCur = regions[baseIdx].size();
					std::vector<std::vector<double> > pointDataCur(pointNumCur);
					for (int j = 0; j < pointNumCur; ++j)
					{
						pointDataCur[j].resize(3);
						pointDataCur[j][0] = this->pointData.pts[regions[baseIdx][j]].x;
						pointDataCur[j][1] = this->pointData.pts[regions[baseIdx][j]].y;
						pointDataCur[j][2] = this->pointData.pts[regions[baseIdx][j]].z;
					}

					PCAFunctions pcaer;
					pcaer.PCASingle(pointDataCur, patches[baseIdx]);

					// 更新后重置索引，重新开始遍历
					i = 0;
					continue;
				}
			}
			// 如果没有更新，继续处理下一个平面
			++i;
		}	
	}

	regions.erase(std::remove_if(regions.begin(), regions.end(),
		[](const std::vector<int>& region) { return region.empty(); }),
		regions.end());
//------------------------------------------------------------------------

	//// 对每一个 region[i] 作为基准依次合并其他 regions
	//for (size_t baseIdx = 0; baseIdx < regions.size(); ++baseIdx) {
	//	if (regions[baseIdx].empty()) continue;

	//	std::vector<int> mergeIndex;
	//	// 1. 获取 regions[baseIdx] 的最大和最小 x、y 值
	//	float minX = projectPointData.pts[regions[baseIdx][0]].x;
	//	float maxX = projectPointData.pts[regions[baseIdx][0]].x;
	//	float minY = projectPointData.pts[regions[baseIdx][0]].y;
	//	float maxY = projectPointData.pts[regions[baseIdx][0]].y;

	//	for (int idx : regions[baseIdx]) {
	//		minX = std::min<float>(minX, projectPointData.pts[idx].x);
	//		maxX = std::max<float>(maxX, projectPointData.pts[idx].x);
	//		minY = std::min<float>(minY, projectPointData.pts[idx].y);
	//		maxY = std::max<float>(maxY, projectPointData.pts[idx].y);

	//	}

	//	// 2. 遍历其他 regions[i]，从 baseIdx + 1 开始
	//	for (size_t i = baseIdx + 1; i < regions.size(); ++i) {
	//		if (regions[i].empty()) continue;

	//		int innerPointsCount = 0;
	//		for (int idx : regions[i]) {
	//			cv::Point p(static_cast<float>(projectPointData.pts[idx].x), static_cast<float>(projectPointData.pts[idx].y));
	//			// 判断是否在区域内或距离在阈值范围内
	//			bool isInside = (p.x >= minX && p.x <= maxX && p.y >= minY && p.y <= maxY);
	//			if (!isInside) {
	//				// 计算点与区域边界的最小距离
	//				float minDistX = std::min(fabs(p.x - minX), fabs(p.x - maxX));
	//				float minDistY = std::min(fabs(p.y - minY), fabs(p.y - maxY));
	//				float distToRegion = sqrt(minDistX * minDistX + minDistY * minDistY);
	//				isInside = (distToRegion <= 0.1);
	//			}

	//			if (isInside) {
	//				innerPointsCount++;
	//			}
	//		}

	//		// 3. 如果内点数量占 regions[i] 的 10%，则合并
	//		if (innerPointsCount >= regions[i].size() * 0.1) {

	//			mergeIndex.push_back(i);
	//	
	//		}
	//	}

	//	for (auto index: mergeIndex)
	//	{
	//		// 将 regions[i] 中的点加入 regions[baseIdx]
	//		regions[baseIdx].insert(regions[baseIdx].end(), regions[index].begin(), regions[index].end());
	//		// 删除 regions[i] 的点
	//		regions[index].clear(); // 不直接删除，只清空，以保持索引一致性
	//	}
	//}

	//// 清理空的 regions
	//regions.erase(std::remove_if(regions.begin(), regions.end(),
	//	[](const std::vector<int>& region) { return region.empty(); }),
	//	regions.end());
	
}


void LineDetection3D::planeBased3DLineDetection( std::vector<std::vector<int> > &regions, std::vector<PLANE> &planes )
{
	double thAngle = 10.0/180.0*CV_PI;
	double thLineLength = 8*this->scale;
	int numPatches = regions.size();

    // step1: fitting 3D plane via PCA
	std::vector<PCAInfo> patches(numPatches);
#pragma omp parallel for
	for ( int i=0; i<numPatches; ++i )
	{
		int pointNumCur = regions[i].size();
		std::vector<std::vector<double> > pointDataCur(pointNumCur);
		for ( int j=0; j<pointNumCur; ++j )
		{
			pointDataCur[j].resize(3);
			pointDataCur[j][0] = this->pointData.pts[regions[i][j]].x;
			pointDataCur[j][1] = this->pointData.pts[regions[i][j]].y;
			pointDataCur[j][2] = this->pointData.pts[regions[i][j]].z;
		}

		PCAFunctions pcaer;
		pcaer.PCASingle( pointDataCur, patches[i] );

		patches[i].idxAll = regions[i];
		for ( int j=0; j<patches[i].idxIn.size(); ++j )
		{
			int idx = patches[i].idxIn[j];
			patches[i].idxIn[j] = regions[i][idx];
		}
	}

	// step2: 3D line detection
	planes.resize(patches.size());
#pragma omp parallel for
	for(int i=0; i<patches.size(); ++i)
	{
		// A. 3D-2D Projection: project the 3d point onto the plane coordinate
		std::vector<cv::Point2d> pts2d;
		std::vector<double> ptScales;
		
		bool initialized = false;
		cv::Mat_<double> vX, vY;
		cv::Mat_<double> planePt = (cv::Mat_<double>(3,1) << patches[i].planePt.val[0], patches[i].planePt.val[1], patches[i].planePt.val[2]);
		cv::Mat_<double> normal  = (cv::Mat_<double>(3,1) << patches[i].normal.val[0], patches[i].normal.val[1], patches[i].normal.val[2]);

		for(int j=0; j<patches[i].idxAll.size(); ++j)
		{
			int id = patches[i].idxAll[j];
			cv::Mat_<double> pt3d = (cv::Mat_<double>(3,1) << pointData.pts[id].x, pointData.pts[id].y, pointData.pts[id].z );

			cv::Mat_<double> v3d = pt3d - planePt;
			cv::Mat_<double> vOrtho = v3d.dot(normal) * normal;
			cv::Mat_<double> vPlane = v3d - vOrtho;
			cv::Mat_<double> ptPlane = planePt + vPlane;

			if(!initialized)
			{
				vX = vPlane * 1.0/(cv::norm(vPlane));
				vY = vX.cross(normal);
				vY = vY * 1.0/cv::norm(vY);
				initialized = true;
			}
			if( initialized )
			{
				double x = vPlane.dot(vX);
				double y = vPlane.dot(vY);
				pts2d.push_back(cv::Point2d(x,y));
				ptScales.push_back(pcaInfos[id].scale);
			}
		}

		// A. 3D-2D Projection: get the side length of the grid cell
		double gridSideLength = 0;
		std::sort( ptScales.begin(), ptScales.end(), [](const double& lhs, const double& rhs) { return lhs < rhs; } );
		int idxNinety = min( int(double(ptScales.size()) * 0.9), int(ptScales.size()-1) );
		gridSideLength = ptScales[idxNinety] * 0.75;

		// A. 3D-2D Projection: get the binary image of the plane
		double xmin, ymin, xmax, ymax;
		int margin = 0;
		cv::Mat mask;
		bool isok = maskFromPoint( pts2d, gridSideLength, xmin, ymin, xmax, ymax, margin, mask );
		if ( !isok )
		{
			continue;
		}

		// B. 2D Line Detection
		int thLineLengthPixel = max(thLineLength/gridSideLength,10.0);
		std::vector<std::vector<std::vector<cv::Point2d> > > lines2d;
		lineFromMask( mask, thLineLengthPixel, lines2d );
		if (!lines2d.size())
		{
			continue;
		}

		// C. 2D-3D Projection
		planes[i].scale = gridSideLength;
		for ( int m=0; m<lines2d.size(); ++m ) 
		{
			std::vector<std::vector<cv::Point3d> > temp;
			for (int n=0; n<lines2d[m].size(); ++n)
			{
				double length = abs(lines2d[m][n][1].x-lines2d[m][n][0].x) + abs(lines2d[m][n][1].y-lines2d[m][n][0].y);
				if ( length < thLineLengthPixel )
				{
					continue;
				}

				lines2d[m][n][0].x = (lines2d[m][n][0].x - margin) * gridSideLength + xmin;
				lines2d[m][n][0].y = (lines2d[m][n][0].y - margin) * gridSideLength + ymin;

				lines2d[m][n][1].x = (lines2d[m][n][1].x - margin) * gridSideLength + xmin;
				lines2d[m][n][1].y = (lines2d[m][n][1].y - margin) * gridSideLength + ymin;

				cv::Mat_<double> xs = lines2d[m][n][0].x * vX;
				cv::Mat_<double> ys = lines2d[m][n][0].y * vY;
				cv::Mat_<double> pts = planePt + xs + ys;

				cv::Mat_<double> xe = lines2d[m][n][1].x * vX;
				cv::Mat_<double> ye = lines2d[m][n][1].y * vY;
				cv::Mat_<double> pte = planePt + xe + ye;

				std::vector<cv::Point3d> line3dTemp(2);
				line3dTemp[0] = cv::Point3d(pts(0), pts(1), pts(2));
				line3dTemp[1] = cv::Point3d(pte(0), pte(1), pte(2));

				temp.push_back( line3dTemp );
			}
			if (temp.size())
			{
				planes[i].lines3d.push_back(temp);
			}
		}
	}
}

bool LineDetection3D::maskFromPoint( std::vector<cv::Point2d> &pts2d, double radius, double &xmin, double &ymin, double &xmax, double &ymax, int &margin, cv::Mat &mask )
{
	xmin=10000000, ymin = 10000000;
	xmax=-xmin;
	ymax=-ymin;
	for (int i=0; i<pts2d.size(); ++i)
	{
		if(pts2d[i].x < xmin) { xmin = pts2d[i].x; }
		if(pts2d[i].x > xmax) { xmax = pts2d[i].x; }

		if(pts2d[i].y < ymin) { ymin = pts2d[i].y; }
		if(pts2d[i].y > ymax) { ymax = pts2d[i].y; }
	}

	margin = 4;
	int cols = (xmax-xmin) / radius + 2*margin;
	int rows = (ymax-ymin) / radius + 2*margin;
	if ( cols < 10 || rows < 10 )
	{
		return false;
	}

	mask = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));
	for (int i=0; i<pts2d.size(); ++i)
	{
		int xInt = int((pts2d[i].x-xmin)/radius+0.5+margin);
		int yInt = int((pts2d[i].y-ymin)/radius+0.5+margin);
		mask.at<uchar>(yInt,xInt) = 255;
	}
	return true;
}

void LineDetection3D::lineFromMask( cv::Mat &mask, int thLineLengthPixel, std::vector<std::vector<std::vector<cv::Point2d> > > &lines )
{
	lines.clear();

	// get mask image via dilate and erode
	cv::Mat mask2;
	cv::dilate(mask, mask2, cv::Mat());
	cv::erode(mask2, mask2, cv::Mat());

	// A. contours
	double thLength = thLineLengthPixel;
	
	std::vector<std::vector<cv::Point> > contours;  
	std::vector<cv::Vec4i> hierarchy;  
	cv::findContours(mask2, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// B. line fitting from the contours
	for ( int i=0; i<contours.size(); ++i )
	{
		if ( contours[i].size() < 4*thLength  )
		{
			continue;
		}

		std::vector<std::vector<cv::Point2d> > lineTemp;
		LineFunctions::lineFitting( mask2.rows, mask2.cols, contours[i], thLength, lineTemp );
		lines.push_back(lineTemp);
	}
}


void LineDetection3D::postProcessing( std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines )
{
	// step1: plane line regularization
	outliersRemoval( planes );

	// step2: line merging
	lineMerging( planes, lines );
}

void LineDetection3D::outliersRemoval( std::vector<PLANE> &planes )
{
	double thCosAngleIN = cos(12.5/180.0*CV_PI);
	double thCosAngleNEW = cos(30.0/180.0*CV_PI);
	double thNonStructPlaneRatio = 0.3;
	double thAngle = 12.5;
	double thCosAngleParal = cos(thAngle/180.0*CV_PI);
	double thCosAngleOrtho = cos((90.0-thAngle)/180.0*CV_PI);
	double thNonStructLineRatio = 10;
	double thStructPlane = 60*this->scale;

	std::vector<int> isPlaneGood(planes.size(), 0);
#pragma omp parallel for
	for (int i=0; i<planes.size(); ++i)
	{
		if (!planes[i].lines3d.size())
		{
			continue;
		}

		// step1: remove non-structural planes
		std::vector<double> lengthsAll;
		std::vector<cv::Mat> orientsAll;
		std::vector<std::pair<int, double> > lineInfos;
		std::vector<std::vector<double> > lengths(planes[i].lines3d.size());
		std::vector<std::vector<cv::Mat> > orients(planes[i].lines3d.size());

		double totalLength = 0.0;
		int count = 0;
		for (int m=0; m<planes[i].lines3d.size(); ++m)
		{
			lengths[m].resize(planes[i].lines3d[m].size());
			orients[m].resize(planes[i].lines3d[m].size());
			for (int n=0; n<planes[i].lines3d[m].size(); ++n)
			{
				cv::Mat orientTemp = cv::Mat(planes[i].lines3d[m][n][1] - planes[i].lines3d[m][n][0]);
				double lengthTemp = cv::norm(orientTemp);
				lengthsAll.push_back(lengthTemp);
				lengths[m][n] = lengthTemp;

				orientTemp *= 1.0/lengthTemp;
				orientsAll.push_back(orientTemp);
				orients[m][n] = orientTemp;

				std::pair<int, double> lineInfoTemp(count, lengthTemp);
				lineInfos.push_back(lineInfoTemp);

				totalLength += lengthTemp;
				count ++;
			}
		}
		std::sort( lineInfos.begin(), lineInfos.end(), [](const std::pair<int, double>& lhs, const std::pair<int, double>& rhs) { return lhs.second > rhs.second; } );

		std::vector<cv::Mat> clusterOrient;
		std::vector<std::pair<int, double> > clusterInfos;
		for (int j=0; j<lineInfos.size(); ++j)
		{
			int id = lineInfos[j].first;
			double length = lineInfos[j].second;

			if (!clusterInfos.size())
			{
				clusterInfos.push_back(std::pair<int, double>(clusterInfos.size(), length));
				clusterOrient.push_back(orientsAll[id]);
				continue;
			}

			bool isIn = false;
			double cosValueMin = 100;
			for (int m=0; m<clusterInfos.size(); ++m)
			{
				double cosValue = abs(orientsAll[id].dot(clusterOrient[m]));
				if ( cosValue < cosValueMin )
				{
					cosValueMin =  cosValue;
				}
				if (cosValue > thCosAngleIN)
				{
					clusterInfos[m].second += length;
					isIn = true;
					break;
				}
			}

			if (!isIn && cosValueMin < thCosAngleNEW)
			{
				clusterInfos.push_back(std::pair<int, double>(clusterInfos.size(), length));
				clusterOrient.push_back(orientsAll[id]);
				continue;
			}
		}

		double scaleCur = max(this->scale,planes[i].scale);
		if ( clusterInfos.size() > 1)
		{
			double LStruct =  clusterInfos[0].second + clusterInfos[1].second;
			if( LStruct < thNonStructPlaneRatio*totalLength || LStruct < thStructPlane ) 
			{
				continue;
			}
		}

		// step2: remove non-structural lines
		PLANE planeNew;
		planeNew.scale = planes[i].scale;
		//double scaleCur = planes[i].scale;
		double thNonStructLineLength = scaleCur*thNonStructLineRatio;
		for (int m=0; m<planes[i].lines3d.size(); ++m)
		{
			int numLines = planes[i].lines3d[m].size();

			double lengthTotal = 0.0;
			for (int n=0; n<numLines; ++n)
			{
				lengthTotal += lengths[m][n];
			}

			double ratioStruct = 0.0;
			double lengthStruct = 0.0;
			std::vector<int> isStruct(numLines, 0);
			if (numLines > 1)
			{
				// judge if the contour is structural
				std::vector<int> idxOrthoPara;
				for (int n=0; n<numLines-1; ++n)
				{
					int id1 = n;
					int id2 = (n+1)%numLines;

					double cosAngle = abs(orients[m][id1].dot(orients[m][id2]));
					if (cosAngle > thCosAngleParal || cosAngle < thCosAngleOrtho)
					{
						idxOrthoPara.push_back(id1);
						idxOrthoPara.push_back(id2);
					}
				}

				if (idxOrthoPara.size())
				{
					// structural ratio
					std::sort( idxOrthoPara.begin(), idxOrthoPara.end(), [](const int& lhs, const int& rhs) { return lhs > rhs; } );

					int idTemp = idxOrthoPara[0];
					isStruct[idTemp] = 1;
					lengthStruct = lengths[m][idTemp];
					for (int n=0; n<idxOrthoPara.size(); ++n)
					{
						if (idxOrthoPara[n] != idTemp)
						{
							lengthStruct += lengths[m][idxOrthoPara[n]];
							idTemp = idxOrthoPara[n];
							isStruct[idTemp] = 1;
						}
					}

					ratioStruct = lengthStruct/lengthTotal;
				}
			}

			std::vector<std::vector<cv::Point3d> > contourTemp;
			for (int n=0; n<numLines; ++n)
			{
				double thLengthTemp = 0.0;
				if (isStruct[n])
				{
					if(ratioStruct>=0.75) 
					{
						thLengthTemp = thNonStructLineLength;
					}
					else if (ratioStruct>=0.5) 
					{
						thLengthTemp = 2*thNonStructLineLength;
					}
					else 
					{
						thLengthTemp = 4*thNonStructLineLength;
					}
				}
				else
				{
					thLengthTemp = 4*thNonStructLineLength;
				}

				if (lengths[m][n] > thLengthTemp)
				{
					contourTemp.push_back(planes[i].lines3d[m][n]);
				}
			}
			if (contourTemp.size())
			{
				planeNew.lines3d.push_back(contourTemp);
			}
		}

		if (planeNew.lines3d.size())
		{
			planes[i] = planeNew;
			isPlaneGood[i] = 1;
		}
	}

	//
	std::vector<PLANE> planesNew;
	for (int i=0; i<isPlaneGood.size(); ++i)
	{
		if (isPlaneGood[i])
		{
			planesNew.push_back(planes[i]);
		}
	}
	planes = planesNew;
}

void LineDetection3D::lineMerging( std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines )
{
	double thGapRatio = 20;
	double thMergeRatio = 6;
	double thDisHyps = 0.1;

	// get all the lines
	std::vector<double> lineScales;
	for (int i=0; i<planes.size(); ++i)
	{
		for (int m=0; m<planes[i].lines3d.size(); ++m)
		{
			for (int n=0; n<planes[i].lines3d[m].size(); ++n)
			{
				lines.push_back(planes[i].lines3d[m][n]);
				lineScales.push_back(planes[i].scale);
			}
		}
	}

	// get the parameters of each 3d line
	std::vector<std::vector<double> > lineParas(lines.size()) ;
	std::vector<std::pair<int, double> > lineInfos(lines.size());
	for ( int i=0; i<lines.size(); ++i )
	{
		cv::Mat v(lines[i][1]-lines[i][0]);
		double length = cv::norm(v);
		v *= 1.0/length;

		cv::Mat ptmid((lines[i][1]+lines[i][0])*0.5);
		cv::Mat d = v.cross(ptmid)*(1.0/this->magnitd);

		// get the latitude of the line, longitude is not stable
		double latitude = asin(abs(v.at<double>(2)));

		// the length of the line
		lineParas[i].resize(6);
		lineParas[i][0] = v.at<double>(0);       lineParas[i][1] = v.at<double>(1);       lineParas[i][2] = v.at<double>(2);
		lineParas[i][3] = latitude;   
		lineParas[i][4] = cv::norm(d); 
		lineParas[i][5] = length; 

		lineInfos[i] = std::pair<int,double>(i, length);
	}
	std::sort( lineInfos.begin(), lineInfos.end(), [](const std::pair<int,double>& lhs, const std::pair<int,double>& rhs) { return lhs.second > rhs.second; } );

	// build grid with latitude
	double precision = 6.0/180.0*CV_PI;
	int laSize = CV_PI/2.0/precision;
	std::vector<std::vector<int > > grid(laSize);
	std::vector<int> gridIndex(lineParas.size());
	for ( int i=0; i<lineParas.size(); ++i )
	{
		int la = lineParas[i][3]/precision;
		grid[la].push_back(i);
		gridIndex[i] = la;
	}

	// line merging
	std::vector<bool> isUsed(lines.size(), 0);
	std::vector<std::vector<cv::Point3d> > linesNew;
	for ( int i=0; i<lineInfos.size(); ++i )
	{
		int id0 = lineInfos[i].first;
		if ( isUsed[id0] )
		{
			continue;
		}
		isUsed[id0] = 1;

		double lineScale = max(lineScales[id0], this->scale);
		double vx0 = lineParas[id0][0], vy0 = lineParas[id0][1], vz0 = lineParas[id0][2];
		double d0 = lineParas[id0][4], length0 = lineParas[id0][5];
		cv::Point3d pts0 = lines[id0][0], pte0 = lines[id0][1];

		// get the merging hypotheses
		std::vector<int> idHyps;
		for (int j=-1; j<=1; ++j)
		{
			int latemp = gridIndex[id0]+j;
			int la = (latemp+laSize)%laSize;
			for ( int m=0; m<grid[la].size(); ++m )
			{
				int idTemp = grid[la][m];
				if (abs(lineParas[idTemp][4]-d0) < thDisHyps)
				{
					idHyps.push_back(idTemp);
				}
			}
		}

		// try merging
		for (int j=0; j<idHyps.size(); ++j)
		{
			int id1 = idHyps[j];
			if ( isUsed[id1] )
			{
				continue;
			}

			cv::Point3d pts1 = lines[id1][0], pte1 = lines[id1][1];
			double length1 = lineParas[id1][5];

			// judge the distance between two line
			cv::Point3d v1 = pts0 - pts1;
			double disNormal1 = v1.x*vx0 + v1.y*vy0 + v1.z*vz0;
			cv::Point3d vOrtho1 = v1 - disNormal1*cv::Point3d(vx0, vy0, vz0);
			double disOrtho1 = sqrt(vOrtho1.x*vOrtho1.x + vOrtho1.y*vOrtho1.y + vOrtho1.z*vOrtho1.z);

			cv::Point3d v2 = pts0 - pte1;
			double disNormal2 = v2.x*vx0 + v2.y*vy0 + v2.z*vz0;
			cv::Point3d vOrtho2 = v2 - disNormal2*cv::Point3d(vx0, vy0, vz0);
			double disOrtho2 = sqrt(vOrtho2.x*vOrtho2.x + vOrtho2.y*vOrtho2.y + vOrtho2.z*vOrtho2.z);

			if ( disOrtho1 > thMergeRatio*lineScale || disOrtho2 > thMergeRatio*lineScale )
			{
				continue;
			}

			// judge the overlapping ratio of two line
			cv::Point3d d1 = pts0 - pts1, d2 = pts0 - pte1, d3 = pte0 - pts1, d4 = pte0 - pte1;
			double dis1 = sqrt(d1.x*d1.x + d1.y*d1.y + d1.z*d1.z);
			double dis2 = sqrt(d2.x*d2.x + d2.y*d2.y + d2.z*d2.z);
			double dis3 = sqrt(d3.x*d3.x + d3.y*d3.y + d3.z*d3.z);
			double dis4 = sqrt(d4.x*d4.x + d4.y*d4.y + d4.z*d4.z);
			double disMerge = max( max(dis1, dis2), max(dis3, dis4) );

			double gapLength = disMerge - length0 - length1;
			double gapRatio = gapLength / length0;
			if ( gapRatio < 0.1 && gapLength < thGapRatio*lineScale )
			{
				// update line id0
				if (gapRatio > 0)
				{
					if (dis1 == disMerge)
					{
						double disNormal = d1.x*vx0 + d1.y*vy0 + d1.z*vz0;
						lines[id0][1] = pts0 - disNormal*cv::Point3d(vx0, vy0, vz0);
					}
					else if (dis2 == disMerge)
					{
						double disNormal = d2.x*vx0 + d2.y*vy0 + d2.z*vz0;
						lines[id0][1] = pts0 - disNormal*cv::Point3d(vx0, vy0, vz0);
					}
					else if (dis3 == disMerge)
					{
						double disNormal = d3.x*vx0 + d3.y*vy0 + d3.z*vz0;
						lines[id0][0] = pte0 - disNormal*cv::Point3d(vx0, vy0, vz0);
					}
					else
					{
						double disNormal = d4.x*vx0 + d4.y*vy0 + d4.z*vz0;
						lines[id0][0] = pte0 - disNormal*cv::Point3d(vx0, vy0, vz0);
					}
				}

				isUsed[id1] = 1;
			}
		}

		linesNew.push_back(lines[id0]);
	}

	lines = linesNew;
}
