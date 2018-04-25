#pragma once
//ann_bp.h//
#ifndef _ANN_BP_H_
#define _ANN_BP_H_

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <windows.h>
#include <ctime>

class Ann_bp
{
	public:
		explicit Ann_bp(int _SampleN, int nNIL, int nNOL, const int nNHL, float _sR = 0.2);
		~Ann_bp();

		void train(int _sampleNum, float** _trainMat, int** _labelMat);
		void predict(float* in, float* proba);

	private:
		int numNodesInputLayer;
		int numNodesOutputLayer;
		int numNodesHiddenLayer;
		int SampleCount;               //总的训练样本数
		double ***weights;            //网络权值
		double **bias;                 //网络偏置
		float studyRate;               //学习速率

		double *hidenLayerOutput;     //隐藏层各结点的输出值
		double *outputLayerOutput;     //输出层各结点的输出值

		double ***allDeltaBias;        //所有样本的偏置更新量
		double ****allDeltaWeights;    //所有样本的权值更新量
		double **outputMat;            //所有样本的输出层输出

		void train_vec(const float* _trainVec, const int* _labelVec, int index);
		double sigmoid(double x) { return 1 / (1 + exp(-1 * x)); }
		bool isNotConver(const int _sampleNum, int** _labelMat, double _thresh);
		// bool Ann_bp::isNotConver(const int _sampleNum, int** _labelMat, double _thresh);

};

#endif
