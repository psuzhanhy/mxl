#ifndef PARAM_H
#define PARAM_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "matvec.h"


struct MxLParam
{
	int numClass;
	int dimension;
	MxLParam(int numclass, int dim): numClass(numclass), dimension(dim) {} 
	virtual ~MxLParam(){};
};

struct BlockCholeskey: public MxLParam
{
	std::vector<CSR_matrix> factorArray;
	BlockCholeskey(int numclass, int dim, bool zeroinit); 
	~BlockCholeskey(){}
	BlockCholeskey& operator+= (BlockCholeskey const& bcholRHS);
	bool operator== (BlockCholeskey const& bcholRHS);
	
};

struct ClassMeans: public MxLParam
{
	std::vector<std::vector<double> > meanVectors;
    ClassMeans(int numclass, int dim, bool zeroinit);
	~ClassMeans() {}	
	ClassMeans& operator+= (ClassMeans const& clmsRHS);
};

#endif
