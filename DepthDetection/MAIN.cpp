#include "stdafx.h"
#include "FeatureMatch.h"
#include "SurfMatch.h"

#define TEST2

#ifdef TEST1
int main()
{
	SurfMatch A;
	A.inputImage();
	A.surfMatch();
	A.RANSACMatch();
}
#elif defined TEST2
int main()
{
	FeatureMatch matcher;
	matcher.inputImage();
	matcher.extractFeature();
	matcher.matchFeatures();
	matcher.initStructure();
}
#endif