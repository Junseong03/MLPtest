#pragma once
class CMLP
{
public:
	CMLP();
	~CMLP();
	#define LEARNING_RATE 0.1
	int m_iNumInNodes;		// �Է³���� ��
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;	// hidden only
	int m_iNumTotalLayer;	// inputLayer + hiddenLayer + outputLayer
	int* m_NumNodes;		// [0]=input node, [1..]-hiddenLayer, [m_iNumHiddenLayer + 1], output Layer, ����

	double*** m_Weight;		// [���� Layer][���� ���][���� ���]
	double** m_NodeOut;	// [Layer][Node]

	double* pInValue, * pOutValue;	// �Է·��̾�, ��·��̾�
	double* pCorrectOutValue;		// ���䷹�̾�
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