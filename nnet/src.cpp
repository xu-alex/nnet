// nnet.cpp: 定义控制台应用程序的入口点。
// https://blog.csdn.net/ap1005834/article/details/52951501

#include "stdafx.h"
#include "ann_bp.h"

using std::cout;
using std::endl;

int main()
{
	const int hidnodes = 8;   // 单层隐藏层的结点数
	const int inNodes = 10;   // 输入层结点数
	const int outNodes = 5;   // 输出层结点数

	const int trainClass = 5; // 5个类别
	const int numPerClass = 30;  // 每个类别30个样本点

	int sampleN = trainClass * numPerClass;     // 每类训练样本数为30，5个类别，总的样本数为150
	// char* p=new char[6]; 这是一个new的使用方法
	float **trainMat = new float* [sampleN];     // 定义一个容量为150的指向指针的指针，用于存放训练集数据的输入部分--trainMat
	// 按行存储所有样本的输入部分，每一行表示一个样本的输入
	for (int k = 0; k < trainClass; ++k) 
	{
		for (int i = k * numPerClass; i < (k + 1) * numPerClass; ++i)  
		{
			trainMat[i] = new float[inNodes];
			for (int j = 0; j < inNodes; ++j) 
				trainMat[i][j] = rand() % 1000 / 10000.0 + 0.1*(2 * k + 1);
		}
	}

	// 按行存储所有样本的输出部分，每一行表示一个样本的输出部分
	int **labelMat = new int*[sampleN]; // 定义一个容量为150的指向指针的指针，用于存放训练集数据的输出部分--labelMat
	for (int k = 0; k < trainClass; ++k) 
	{
		for (int i = k * numPerClass; i < (k + 1) * numPerClass; ++i) 
		{
			labelMat[i] = new int[outNodes];
			for (int j = 0; j <outNodes; ++j) //因为outNodes的值和trainClass的值相同
			{
				if (j == k)
					labelMat[i][j] = 1;
				else
					labelMat[i][j] = 0;
			}
		}
	}

	Ann_bp ann_classify(sampleN, inNodes, outNodes, hidnodes, 0.12);  // 定义一个Ann_bp类，其初始化参数为：输入层为10个结点，输出层5个结点，单层隐藏层
	ann_classify.train(sampleN, trainMat, labelMat); // 调用ann_classify的成员函数train来训练数据集


	for (int i = 0; i < 30; ++i) 
	{
		ann_classify.predict(trainMat[i + 120]);
		cout << endl;
	}

	//释放内存
	for (int i = 0; i < sampleN; ++i)
		delete[] trainMat[i];
	delete[] trainMat;

	for (int i = 0; i < sampleN; ++i)
		delete[] labelMat[i];
	delete[] labelMat;

	system("pause");
	return 0;
}
