#include "stdafx.h"
#include "FeatureMatch.h"

int main()
{
	FeatureMatch A;
	A.inputImage();
	A.surfMatch();
	A.RANSACMatch();
}