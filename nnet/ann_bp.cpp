//ann_bp.cpp//
#include "stdafx.h"
#include "ann_bp.h"
#include <math.h>

using std::cout;
using std::endl;

// ���幹�캯��
Ann_bp::Ann_bp(int _SampleN, int nNIL, int nNOL, const int nNHL, float _sR) :
	SampleCount(_SampleN), numNodesInputLayer(nNIL), numNodesOutputLayer(nNOL),
	numNodesHiddenLayer(nNHL), studyRate(_sR)
{
	// ����Ȩֵ�ռ�,����ʼ��
	srand((unsigned)time(NULL)); // ��ʱ��Ϊ�ں����һ�����������
	weights = new double**[2];  // ����Ȩֵ����--weights,��������ά��Ϊ2��ÿ��Ԫ�ض���ָ��double����ָ���ָ��
	weights[0] = new double* [numNodesInputLayer]; // ����������Ȩֵ����ÿ��Ԫ�ض���ָ��double����ָ��
	for (int i = 0; i < numNodesInputLayer; ++i)   // �������ÿ��������
	{
		weights[0][i] = new double[numNodesHiddenLayer];
		for (int j = 0; j <numNodesHiddenLayer; ++j) 
		{
			// rand()��ʾȡ�������rand() % (2000)��ʾȡ0��1999֮���ֵ��������ʾ��-1��1֮��ȡֵ��
			weights[0][i][j] = (rand() % (2000) / 1000.0 - 1); // ����ÿ������ڵ㵽ÿ�����ز�ڵ��Ȩֵ
		}
	}
	weights[1] = new double* [numNodesHiddenLayer]; // �������ز��Ȩֵ����
	for (int i = 0; i < numNodesHiddenLayer; ++i)  // �����ȡÿ�����ز�Ľڵ�
	{
		weights[1][i] = new double[numNodesOutputLayer];
		for (int j = 0; j < numNodesOutputLayer; ++j) 
		{
			weights[1][i][j] = (rand() % (2000) / 1000.0 - 1); // ����ÿ�����ز�ڵ㵽ÿ�������ڵ��Ȩֵ
		}
	}

	// ����ƫ�ÿռ䣬����ʼ��
	bias = new double *[2];
	bias[0] = new double[numNodesHiddenLayer];   // ���ز�ÿ���ڵ��ƫ��ֵ
	for (int i = 0; i < numNodesHiddenLayer; ++i) 
	{
		bias[0][i] = (rand() % (2000) / 1000.0 - 1);
	}
	bias[1] = new double[numNodesOutputLayer];  // �����ÿ���ڵ��ƫ��ֵ
	for (int i = 0; i < numNodesOutputLayer; ++i) 
	{
		bias[1][i] = (rand() % (2000) / 1000.0 - 1); //-1��1֮��
	}

	// �������ز���������ֵ�ռ�
	hidenLayerOutput = new double[numNodesHiddenLayer];
	// �����������������ֵ�ռ�
	outputLayerOutput = new double[numNodesOutputLayer];

	// ��������������Ȩֵ�������洢�ռ�
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

	// ��������������ƫ�ø������洢�ռ�
	allDeltaBias = new double **[_SampleN];
	for (int k = 0; k < _SampleN; ++k) 
	{
		allDeltaBias[k] = new double *[2];
		allDeltaBias[k][0] = new double[numNodesHiddenLayer];
		allDeltaBias[k][1] = new double[numNodesOutputLayer];
	}

	// �����洢�������������������ռ�
	outputMat = new double*[_SampleN];
	for (int k = 0; k < _SampleN; ++k) 
		outputMat[k] = new double[numNodesOutputLayer];
}

// ������������
Ann_bp::~Ann_bp()
{
	//�ͷ�Ȩֵ�ռ�
	for (int i = 0; i < numNodesInputLayer; ++i)
		delete[] weights[0][i];
	for (int i = 1; i < numNodesHiddenLayer; ++i)
		delete[] weights[1][i];
	for (int i = 0; i < 2; ++i)
		delete[] weights[i];
	delete[] weights;

	//�ͷ�ƫ�ÿռ�
	for (int i = 0; i < 2; ++i)
		delete[] bias[i];
	delete[] bias;

	//�ͷ�����������Ȩֵ�������洢�ռ�
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

	//�ͷ�����������ƫ�ø������洢�ռ�
	for (int k = 0; k < SampleCount; ++k) 
	{
		for (int i = 0; i < 2; ++i)
			delete[] allDeltaBias[k][i];
		delete[] allDeltaBias[k];
	}
	delete[] allDeltaBias;

	//�ͷŴ洢�������������������ռ�
	for (int k = 0; k < SampleCount; ++k)
		delete[] outputMat[k];
	delete[] outputMat;

}

void Ann_bp::train(const int _sampleNum, float** _trainMat, int** _labelMat)
{
	/*
	_sampleNum��ʾ��������
	_trainMat��ʾ����������ֵ�ľ���
	_labelMat��ʾ���������ֵ�ľ���
	*/

	double thre = 1e-4;  // ����һ����ֵ����ʾѵ���ľ���
	int tt = 0;
	int iter = 100000;   // ����������

	do
	{
		for (int i = 0; i < _sampleNum; ++i)
			train_vec(_trainMat[i], _labelMat[i], i);  // �ֱ����ÿ��������Ӧ������Ȩֵ��ƫ�õĸ�����

		// �������ÿ������������Ȩ�ص�Ӱ��
		for (int index = 0; index < _sampleNum; ++index)
		{
			for (int i = 0; i < numNodesInputLayer; ++i) // ����ÿ������ڵ㵽���ز�ÿ���ڵ��Ȩֵ
			{
				for (int j = 0; j < numNodesHiddenLayer; ++j)
					weights[0][i][j] -= studyRate * allDeltaWeights[index][0][i][j]; // �ݶ��½���
			}
			for (int i = 0; i < numNodesHiddenLayer; ++i) // ����ÿ�����ز�ڵ㵽�����ÿ���ڵ��Ȩֵ
			{
				for (int j = 0; j < numNodesOutputLayer; ++j)
					weights[1][i][j] -= studyRate * allDeltaWeights[index][1][i][j];
			}
		}

		// �������ÿ������������ƫ�õ�Ӱ��
		for (int index = 0; index < _sampleNum; ++index)
		{
			for (int i = 0; i < numNodesHiddenLayer; ++i)
				bias[0][i] -= studyRate * allDeltaBias[index][0][i];

			for (int i = 0; i < numNodesOutputLayer; ++i)
				bias[1][i] -= studyRate * allDeltaBias[index][1][i];
		}

		++tt;
	} while (isNotConver(_sampleNum, _labelMat, thre) && tt< iter);

	cout << "Ȩֵ��ƫ��ѵ���ɹ��ˣ�" << endl;
}

void Ann_bp::train_vec(const float* _trainVec, const int* _labelVec, int index)
{
	/*
		_trainVec��ʾÿ������������ֵ
		_labelVec��ʾÿ�����������ֵ
		index��ʾ���������
	*/

	// ��������ز�������
	for (int i = 0; i < numNodesHiddenLayer; ++i) 
	{
		double z = 0.0;
		for (int j = 0; j < numNodesInputLayer; ++j) 
			z += _trainVec[j] * weights[0][j][i];
		z += bias[0][i];
		hidenLayerOutput[i] = sigmoid(z);
	}

	// ���������ڵ�����ֵ
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

	// �Ӻ���ǰ������ƫ�ü�Ȩ�ظ��������������¡���Ϊ���򴫲��Ĺ��̡�
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
			z += weights[1][j][k] * allDeltaBias[index][1][k]; // ÿ������ڵ�����������ڵ��Ӱ����ۼ�
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
	_sampleNum��ʾ��������
	_labelMat��ʾ���������ֵ�ľ���
	_thresh��ʾ�Ż�����
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

	// �����ڲ�������static����������Ϊ������һ��ͨ�Ż���
	static int tt = 0;
	cout << "��" << ++tt << "��ѵ����" << lossFunc << endl;

	if (lossFunc > _thresh)  // �����ʧ������ֵ������ֵ����ʾ�����㾫��Ҫ���򷵻�һ��trueֵ�����Һ��������ﱻǿ�ƽ�����
		return true; // return�����ں�����ǿ�ƽ�����

	return false;
}

void Ann_bp::predict(float* in)
{
	/*
	in��ʾ��������
	*/

	// ��������ز�������
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

	// ���������ڵ�����ֵ
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