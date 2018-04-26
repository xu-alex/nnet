//ann_bp.cpp//
#include "stdafx.h"
#include "ann_bp.h"
#include <math.h>

using std::cout;
using std::endl;

// 定义构造函数
Ann_bp::Ann_bp(int _SampleN, int nNIL, int nNOL, const int nNHL, float _sR) :
	SampleCount(_SampleN), numNodesInputLayer(nNIL), numNodesOutputLayer(nNOL),
	numNodesHiddenLayer(nNHL), studyRate(_sR)
{
	// 创建权值空间,并初始化
	srand((unsigned)time(NULL)); // 以时间为内核设计一个随机数种子
	weights = new double**[2];  // 定义权值矩阵--weights,该向量的维度为2，每个元素都是指向double量的指针的指针
	weights[0] = new double* [numNodesInputLayer]; // 定义输入层的权值矩阵，每个元素都是指向double量的指针
	for (int i = 0; i < numNodesInputLayer; ++i)   // 逐个处理每个输入结点
	{
		weights[0][i] = new double[numNodesHiddenLayer];
		for (int j = 0; j <numNodesHiddenLayer; ++j) 
		{
			// rand()表示取随机数；rand() % (2000)表示取0到1999之间的值；该语句表示在-1到1之间取值。
			weights[0][i][j] = (rand() % (2000) / 1000.0 - 1); // 这是每个输入节点到每个隐藏层节点的权值
		}
	}
	weights[1] = new double* [numNodesHiddenLayer]; // 定义隐藏层的权值矩阵
	for (int i = 0; i < numNodesHiddenLayer; ++i)  // 逐个读取每个隐藏层的节点
	{
		weights[1][i] = new double[numNodesOutputLayer];
		for (int j = 0; j < numNodesOutputLayer; ++j) 
		{
			weights[1][i][j] = (rand() % (2000) / 1000.0 - 1); // 这是每个隐藏层节点到每个输出层节点的权值
		}
	}

	// 创建偏置空间，并初始化
	bias = new double *[2];
	bias[0] = new double[numNodesHiddenLayer];   // 隐藏层每个节点的偏置值
	for (int i = 0; i < numNodesHiddenLayer; ++i) 
	{
		bias[0][i] = (rand() % (2000) / 1000.0 - 1);
	}
	bias[1] = new double[numNodesOutputLayer];  // 输出层每个节点的偏置值
	for (int i = 0; i < numNodesOutputLayer; ++i) 
	{
		bias[1][i] = (rand() % (2000) / 1000.0 - 1); //-1到1之间
	}

	// 创建隐藏层各结点的输出值空间
	hidenLayerOutput = new double[numNodesHiddenLayer];
	// 创建输出层各结点的输出值空间
	outputLayerOutput = new double[numNodesOutputLayer];

	// 创建所有样本的权值更新量存储空间
	allDeltaWeights = new double ***[_SampleN];
	for (int k = 0; k < _SampleN; ++k) 
	{
		allDeltaWeights[k] = new double**[2];
		allDeltaWeights[k][0] = new double *[numNodesInputLayer];
		for (int i = 0; i < numNodesInputLayer; ++i) 
			allDeltaWeights[k][0][i] = new double[numNodesHiddenLayer];
		allDeltaWeights[k][1] = new double *[numNodesHiddenLayer];
		for (int i = 0; i < numNodesHiddenLayer; ++i) 
			allDeltaWeights[k][1][i] = new double[numNodesOutputLayer];
	}

	// 创建所有样本的偏置更新量存储空间
	allDeltaBias = new double **[_SampleN];
	for (int k = 0; k < _SampleN; ++k) 
	{
		allDeltaBias[k] = new double *[2];
		allDeltaBias[k][0] = new double[numNodesHiddenLayer];
		allDeltaBias[k][1] = new double[numNodesOutputLayer];
	}

	// 创建存储所有样本的输出层输出空间
	outputMat = new double*[_SampleN];
	for (int k = 0; k < _SampleN; ++k) 
		outputMat[k] = new double[numNodesOutputLayer];
}

// 定义析构函数
Ann_bp::~Ann_bp()
{
	//释放权值空间
	for (int i = 0; i < numNodesInputLayer; ++i)
		delete[] weights[0][i];
	for (int i = 1; i < numNodesHiddenLayer; ++i)
		delete[] weights[1][i];
	for (int i = 0; i < 2; ++i)
		delete[] weights[i];
	delete[] weights;

	//释放偏置空间
	for (int i = 0; i < 2; ++i)
		delete[] bias[i];
	delete[] bias;

	//释放所有样本的权值更新量存储空间
	for (int k = 0; k < SampleCount; ++k) 
	{
		for (int i = 0; i < numNodesInputLayer; ++i)
			delete[] allDeltaWeights[k][0][i];
		for (int i = 1; i < numNodesHiddenLayer; ++i)
			delete[] allDeltaWeights[k][1][i];
		for (int i = 0; i < 2; ++i)
			delete[] allDeltaWeights[k][i];
		delete[] allDeltaWeights[k];
	}
	delete[] allDeltaWeights;

	//释放所有样本的偏置更新量存储空间
	for (int k = 0; k < SampleCount; ++k) 
	{
		for (int i = 0; i < 2; ++i)
			delete[] allDeltaBias[k][i];
		delete[] allDeltaBias[k];
	}
	delete[] allDeltaBias;

	//释放存储所有样本的输出层输出空间
	for (int k = 0; k < SampleCount; ++k)
		delete[] outputMat[k];
	delete[] outputMat;

}

void Ann_bp::train(const int _sampleNum, float** _trainMat, int** _labelMat)
{
	/*
	_sampleNum表示样本数量
	_trainMat表示样本的输入值的矩阵
	_labelMat表示样本的输出值的矩阵
	*/

	double thre = 1e-4;  // 定义一个阈值，表示训练的精度
	int tt = 0;
	int iter = 100000;   // 最大迭代次数

	do
	{
		for (int i = 0; i < _sampleNum; ++i)
			train_vec(_trainMat[i], _labelMat[i], i);  // 分别计算每个样本对应的所有权值和偏置的更新量

		// 逐个计算每个样本对所有权重的影响
		for (int index = 0; index < _sampleNum; ++index)
		{
			for (int i = 0; i < numNodesInputLayer; ++i) // 更新每个输入节点到隐藏层每个节点的权值
			{
				for (int j = 0; j < numNodesHiddenLayer; ++j)
					weights[0][i][j] -= studyRate * allDeltaWeights[index][0][i][j]; // 梯度下降法
			}
			for (int i = 0; i < numNodesHiddenLayer; ++i) // 更新每个隐藏层节点到输出层每个节点的权值
			{
				for (int j = 0; j < numNodesOutputLayer; ++j)
					weights[1][i][j] -= studyRate * allDeltaWeights[index][1][i][j];
			}
		}

		// 逐个计算每个样本对所有偏置的影响
		for (int index = 0; index < _sampleNum; ++index)
		{
			for (int i = 0; i < numNodesHiddenLayer; ++i)
				bias[0][i] -= studyRate * allDeltaBias[index][0][i];

			for (int i = 0; i < numNodesOutputLayer; ++i)
				bias[1][i] -= studyRate * allDeltaBias[index][1][i];
		}

		++tt;
	} while (isNotConver(_sampleNum, _labelMat, thre) && tt< iter);

	cout << "权值和偏置训练成功了！" << endl;
}

void Ann_bp::train_vec(const float* _trainVec, const int* _labelVec, int index)
{
	/*
		_trainVec表示每个样本的输入值
		_labelVec表示每个样本的输出值
		index表示样本的序号
	*/

	// 计算各隐藏层结点的输出
	for (int i = 0; i < numNodesHiddenLayer; ++i) 
	{
		double z = 0.0;
		for (int j = 0; j < numNodesInputLayer; ++j) 
			z += _trainVec[j] * weights[0][j][i];
		z += bias[0][i];
		hidenLayerOutput[i] = sigmoid(z);
	}

	// 计算输出层节点的输出值
	for (int i = 0; i < numNodesOutputLayer; ++i) 
	{
		double z = 0.0;
		for (int j = 0; j < numNodesHiddenLayer; ++j) 
		{
			z += hidenLayerOutput[j] * weights[1][j][i];
		}
		z += bias[1][i];
		outputLayerOutput[i] = sigmoid(z);
		outputMat[index][i] = outputLayerOutput[i];
	}

	// 从后向前，计算偏置及权重更新量，但不更新。即为反向传播的过程。
	for (int j = 0; j <numNodesOutputLayer; ++j) 
	{
		allDeltaBias[index][1][j] = (-0.1)*(_labelVec[j] - outputLayerOutput[j])*outputLayerOutput[j]*(1 - outputLayerOutput[j]);
		for (int i = 0; i < numNodesHiddenLayer; ++i) 
		{
			allDeltaWeights[index][1][i][j] = allDeltaBias[index][1][j] * hidenLayerOutput[i];
		}
	}
	for (int j = 0; j < numNodesHiddenLayer; ++j) 
	{
		double z = 0.0;
		for (int k = 0; k < numNodesOutputLayer; ++k) 
		{
			z += weights[1][j][k] * allDeltaBias[index][1][k]; // 每个隐层节点对所有输出层节点的影响的累加
		}
		allDeltaBias[index][0][j] = z * hidenLayerOutput[j] * (1 - hidenLayerOutput[j]);
		for (int i = 0; i < numNodesInputLayer; ++i) 
		{
			allDeltaWeights[index][0][i][j] = allDeltaBias[index][0][j] * _trainVec[i];
		}
	}

}


bool Ann_bp::isNotConver(const int _sampleNum,int** _labelMat, double _thresh)
{
	/*
	_sampleNum表示样本数量
	_labelMat表示样本的输出值的矩阵
	_thresh表示优化精度
	*/

	double lossFunc = 0.0;
	for (int k = 0; k < _sampleNum; ++k) 
	{
		double loss = 0.0;
		for (int t = 0; t < numNodesOutputLayer; ++t) 
		{
			loss += (outputMat[k][t] - _labelMat[k][t])*(outputMat[k][t] - _labelMat[k][t]);
		}
		lossFunc += (1.0 / 2)*loss;
	}

	lossFunc = lossFunc / _sampleNum;

	// 函数内部声明的static变量，可作为对象间的一种通信机制
	static int tt = 0;
	cout << "第" << ++tt << "次训练：" << lossFunc << endl;

	if (lossFunc > _thresh)  // 如果损失函数的值大于阈值，表示不满足精度要求，则返回一个true值，并且函数在这里被强制结束。
		return true; // return可用于函数的强制结束。

	return false;
}

void Ann_bp::predict(float* in)
{
	/*
	in表示网络输入
	*/

	// 计算各隐藏层结点的输出
	for (int i = 0; i < numNodesHiddenLayer; ++i) 
	{
		double z = 0.0;
		for (int j = 0; j < numNodesInputLayer; ++j) 
		{
			z += in[j] * weights[0][j][i];
		}
		z += bias[0][i];
		hidenLayerOutput[i] = sigmoid(z);
	}

	// 计算输出层节点的输出值
	for (int i = 0; i < numNodesOutputLayer; ++i) 
	{
		double z = 0.0;
		for (int j = 0; j < numNodesHiddenLayer; ++j) 
		{
			z += hidenLayerOutput[j] * weights[1][j][i];
		}
		z += bias[1][i];
		outputLayerOutput[i] = sigmoid(z);
		cout << outputLayerOutput[i] << " ";
	}

}