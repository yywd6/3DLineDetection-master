#ifndef _COMMON_FUNCTIONS_
#define _COMMON_FUNCTIONS_
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "nanoflann.hpp"
#include "utils.h"

using namespace cv;
using namespace std;
using namespace nanoflann;

class LineFunctions
{
public:
	LineFunctions(void){};
	~LineFunctions(void){};

public:
	static void lineFitting( int rows, int cols, std::vector<cv::Point> &contour, double thMinimalLineLength, std::vector<std::vector<cv::Point2d> > &lines );

	static void subDivision( std::vector<std::vector<cv::Point> > &straightString, std::vector<cv::Point> &contour, int first_index, int last_index
		, double min_deviation, int min_size  );

	static void lineFittingSVD( cv::Point *points, int length, std::vector<double> &parameters, double &maxDev );
};

//在 C++ 中，static 关键字用于定义类中的静态成员变量和静态成员函数。
//
//静态成员变量是指在类中定义的静态变量，它不是属于类的任何一个对象，而是属于整个类。因此，所有属于该类的对象都共享同一个静态成员变量。静态成员变量的生命周期从程序开始到程序结束，即使类的所有对象都被销毁了，静态成员变量仍然存在。静态成员变量可以通过类名和作用域运算符（::）来访问，例如 ClassName::static_member。
//
//静态成员函数是指在类中定义的静态函数，它与静态成员变量类似，它不依赖于类的任何一个对象，也不可以访问非静态成员变量。静态成员函数可以通过类名和作用域运算符（::）来调用，例如 ClassName::static_member_function()。
//
//静态成员变量和静态成员函数的作用包括：
//
//可以实现共享数据和共享函数的功能，可以节省内存空间和提高效率；
//可以通过类名来访问和调用，不需要创建对象；
//可以在类外部定义和初始化，但是必须保证其只被初始化一次。
//需要注意的是，静态成员变量和静态成员函数不属于任何一个对象，因此不能使用 this 关键字来访问静态成员变量和静态成员函数。

struct PCAInfo
{
	double lambda0, scale;
	cv::Matx31d normal, planePt,boundMax, boundMin, middlePoints;
	std::vector<int> idxAll, idxIn;

	PCAInfo &operator =(const PCAInfo &info)
	{
		this->lambda0 = info.lambda0;
		this->normal = info.normal;
		this->idxIn = info.idxIn;
		this->idxAll = info.idxAll;
		this->scale = scale;
		return *this;
	}
};

class PCAFunctions 
{
public:
	PCAFunctions(void){};
	~PCAFunctions(void){};

	void Ori_PCA( PointCloud<double> &cloud, int k, std::vector<PCAInfo> &pcaInfos, double &scale, double &magnitd );

	void PCASingle( std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo );

	void MCMD_OutlierRemoval( std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo );

	double meadian( std::vector<double> dataset );
};

#endif //_COMMON_FUNCTIONS_
