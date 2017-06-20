#pragma once
#include "tinydir.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class GlobalMethod
{
public:
	static void getFileNames(string dir_name, vector<string> & names);
};