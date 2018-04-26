#pragma once
#ifndef _ANN_BP_H_
#define _ANN_BP_H_

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <windows.h>
#include <ctime>

// 声明一个类
class Ann_bp
{
	public:
		/*
		  explicit关键字用于修饰只有一个参数的类构造函数, 它的作用是表明该构造函数是显示的, 而非隐式的；
		  explicit关键字只对有一个参数的类构造函数有效, 如果类构造函数参数大于或等于两个时, 是不会产生隐式转换的, explicit也就无效了；
		  也有一个例外，当除了第一个参数以外的其他参数都有默认值的时候, explicit关键字依然有效。
		*/
		// 类构造函数
		explicit Ann_bp(int _SampleN, int nNIL, int nNOL, const int nNHL, float _sR = 0.2);
		// 类析构函数
		~Ann_bp();

		void train(int _sampleNum, float** _trainMat, int** _labelMat);
		void predict(float* in, float* proba);

	private:
		int numNodesInputLayer;
		int numNodesOutputLayer;
		int numNodesHiddenLayer;
		int SampleCount;               // 总的训练样本数
		double ***weights;             // 网络权值
		double **bias;                 // 网络偏置
		float studyRate;               // 学习速率

		double *hidenLayerOutput;      // 隐藏层各结点的输出值
		double *outputLayerOutput;     // 输出层各结点的输出值

		double ***allDeltaBias;        // 所有样本的偏置更新量
		double ****allDeltaWeights;    // 所有样本的权值更新量
		double **outputMat;            // 所有样本的输出层输出

		void train_vec(const float* _trainVec, const int* _labelVec, int index);
		double sigmoid(double x) { return 1 / (1 + exp(-1 * x)); }
		bool isNotConver(const int _sampleNum, int** _labelMat, double _thresh);
};

#endif
