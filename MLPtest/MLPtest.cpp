#include <stdio.h>
#include "MLP.h"

CMLP MultiLayer;

int main(void)
{
	int numofHiddenLayer = 1;
	int HiddenNodes[1] = { 2 };

	MultiLayer.Create(2, HiddenNodes, 1, numofHiddenLayer);

	MultiLayer.m_Weight[0][0][1] = -0.8;
	MultiLayer.m_Weight[0][1][1] = 0.5;
	MultiLayer.m_Weight[0][2][1] = 0.4;

	MultiLayer.m_Weight[0][0][2] = 0.1;
	MultiLayer.m_Weight[0][1][2] = 0.9;
	MultiLayer.m_Weight[0][2][2] = 1.0;

	MultiLayer.m_Weight[1][0][1] = -0.3;
	MultiLayer.m_Weight[1][1][1] = -1.2;
	MultiLayer.m_Weight[1][2][1] = 1.1;


	/*
	double x[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };	// 입력값 셋팅
	
	int n;
	for (n = 0; n < 4; n++)
	{
		//MultiLayer.pInValue[0] = 1; // 바이어스
		MultiLayer.pInValue[1] = x[n][0]; // 입력 전달
		MultiLayer.pInValue[2] = x[n][1];

		MultiLayer.Forward();

		printf("%lf %lf = %lf\n", MultiLayer.pInValue[1], MultiLayer.pInValue[2], MultiLayer.pOutValue[1]);

	}
	printf("\n");
	*/
	MultiLayer.pInValue[1] = 1;
	MultiLayer.pInValue[2] = 1;

	MultiLayer.pCorrectOutValue[1] = 0;

	MultiLayer.Forward();
	MultiLayer.BackPropagationLearning();

	printf("%lf %lf=%lf\n", MultiLayer.pInValue[1],MultiLayer.pInValue[1], MultiLayer.pOutValue[1]);
	return 0;
}