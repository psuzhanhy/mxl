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
	double l2normsq() const;
	BlockCholeskey& operator-= (BlockCholeskey const& bcholRHS);
	BlockCholeskey operator- (BlockCholeskey const &bcholRHS) const;
	BlockCholeskey& operator*= (double scalar);
	bool operator== (BlockCholeskey const& bcholRHS);
	
};

struct ClassMeans: public MxLParam
{
	std::vector<std::vector<double> > meanVectors;
    ClassMeans(int numclass, int dim, bool zeroinit);
	~ClassMeans() {}
	void setzero();
	double l2normsq() const;
	ClassMeans& operator-= (ClassMeans const& clmsRHS);
	ClassMeans operator- (ClassMeans const &clmsRHS) const;
	ClassMeans& operator*= (double scalar);

};



#endif
