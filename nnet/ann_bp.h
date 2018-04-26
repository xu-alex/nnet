#pragma once
#ifndef _ANN_BP_H_
#define _ANN_BP_H_

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <windows.h>
#include <ctime>

// ����һ����
class Ann_bp
{
	public:
		/*
		  explicit�ؼ�����������ֻ��һ���������๹�캯��, ���������Ǳ����ù��캯������ʾ��, ������ʽ�ģ�
		  explicit�ؼ���ֻ����һ���������๹�캯����Ч, ����๹�캯���������ڻ��������ʱ, �ǲ��������ʽת����, explicitҲ����Ч�ˣ�
		  Ҳ��һ�����⣬�����˵�һ�����������������������Ĭ��ֵ��ʱ��, explicit�ؼ�����Ȼ��Ч��
		*/
		// �๹�캯��
		explicit Ann_bp(int _SampleN, int nNIL, int nNOL, const int nNHL, float _sR = 0.2);
		// ����������
		~Ann_bp();

		void train(int _sampleNum, float** _trainMat, int** _labelMat);
		void predict(float* in, float* proba);

	private:
		int numNodesInputLayer;
		int numNodesOutputLayer;
		int numNodesHiddenLayer;
		int SampleCount;               // �ܵ�ѵ��������
		double ***weights;             // ����Ȩֵ
		double **bias;                 // ����ƫ��
		float studyRate;               // ѧϰ����

		double *hidenLayerOutput;      // ���ز���������ֵ
		double *outputLayerOutput;     // �������������ֵ

		double ***allDeltaBias;        // ����������ƫ�ø�����
		double ****allDeltaWeights;    // ����������Ȩֵ������
		double **outputMat;            // ������������������

		void train_vec(const float* _trainVec, const int* _labelVec, int index);
		double sigmoid(double x) { return 1 / (1 + exp(-1 * x)); }
		bool isNotConver(const int _sampleNum, int** _labelMat, double _thresh);
};

#endif
