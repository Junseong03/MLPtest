#pragma once
class CMLP
{
public:
	CMLP();
	~CMLP();
	#define LEARNING_RATE 0.1
	int m_iNumInNodes;		// 입력노드의 수
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;	// hidden only
	int m_iNumTotalLayer;	// inputLayer + hiddenLayer + outputLayer
	int* m_NumNodes;		// [0]=input node, [1..]-hiddenLayer, [m_iNumHiddenLayer + 1], output Layer, 정답

	double*** m_Weight;		// [시작 Layer][시작 노드][연결 노드]
	double** m_NodeOut;	// [Layer][Node]

	double* pInValue, * pOutValue;	// 입력레이어, 출력레이어
	double* pCorrectOutValue;		// 정답레이어
	bool Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer);
private:
	void InitW();
	double ActivationFunc(double weightsum);
public:
	void Forward();
	double** m_ErrorGradient; // [layer][node]
	void BackPropagationLearning();
	bool SaveWeight(char* fname);
};