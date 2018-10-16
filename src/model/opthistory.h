#ifndef OPT_HISTORY_H
#define OPT_HISTORY_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "param.h"

struct OptHistory
{
	int maxIter;
	std::vector<double> fobj;
	std::vector<double> gradNormSq;
	std::vector<double> iterTime;
	std::vector<double> paramChange;
	OptHistory(int iter): maxIter(iter) {}	
};

#endif
