#include "stdafx.h"
#include "DepthDetection.h"

int main()
{
	DepthDetection A;
	A.inputImage();
	A.surfMatch();
	A.RANSACMatch();
}