#ifndef PARAM_H
#define PARAM_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "matvec.h"


struct LogisticParam
{
	int numClass;
	int dimension;
	LogisticParam(int numclass, int dim): numClass(numclass), dimension(dim) {} 
	virtual ~LogisticParam(){};
};

struct BlockCholeskey: public LogisticParam
{
	std::vector<CSR_matrix> factorArray;
	BlockCholeskey(int numclass, int dim, bool zeroinit); 
	~BlockCholeskey(){}
	void setzero();
	double l2normsq() const;
	BlockCholeskey& operator-= (BlockCholeskey const& bcholRHS);
	BlockCholeskey operator- (BlockCholeskey const &bcholRHS) const;
	BlockCholeskey operator+ (BlockCholeskey const &bcholRHS) const;
	BlockCholeskey& operator*= (double scalar);
	BlockCholeskey operator* (double scalar);
	bool operator== (BlockCholeskey const& bcholRHS);
	
};

struct ClassMeans: public LogisticParam
{
	std::vector<std::vector<double> > meanVectors;
    ClassMeans(int numclass, int dim, bool zeroinit);
	~ClassMeans() {}
	void setzero();
	double l2normsq() const;
	ClassMeans& operator-= (ClassMeans const& clmsRHS);
	ClassMeans operator- (ClassMeans const &clmsRHS) const;
	ClassMeans operator+ (ClassMeans const &clmsRHS) const;
	ClassMeans& operator*= (double scalar);
	ClassMeans operator* (double scalar);

};

struct Beta: public LogisticParam
{
	std::vector<std::vector<double>> beta;
	Beta(int numclass, int dim, bool zeroinit);
	~Beta() {}
	Beta& operator-= (Beta const& otherBeta);
	Beta operator- (Beta const &otherBeta) const;
	Beta operator+ (Beta const &otherBeta) const;
	Beta& operator*= (double scalar);
	Beta operator* (double scalar);	
	void setzero();
};


#endif
