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
	void setzero();
	double norm();
	void operator-= (BlockCholeskey const& bcholRHS);
	void operator*= (double scalar);
	bool operator== (BlockCholeskey const& bcholRHS);
	
};

struct ClassMeans: public MxLParam
{
	std::vector<std::vector<double> > meanVectors;
    ClassMeans(int numclass, int dim, bool zeroinit);
	~ClassMeans() {}
	void setzero();
	double norm();
	void operator-= (ClassMeans const& clmsRHS);
	void operator*= (double scalar);

};



#endif
