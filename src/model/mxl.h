#ifndef MXL_H
#define MXL_H 
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "matvec.h"


class MixedLogit 
{
	/*
	*  abstract base class for mixed logit model
	*/
	protected:
		int numClass;
		int dimension;
		CSR_matrix xfeature;
		std::vector<int> label;		

	public:
		MixedLogit(CSR_matrix xf, std::vector<int> lbl, int numclass, int dim): xfeature(xf), label(lbl), numClass(numclass), dimension(dim) {} 
		virtual double negativeLogLik() = 0;
		virtual std::vector<double> multinomialProb(std::vector<double> propensityScore) = 0;
		virtual ~MixedLogit() {}
		 	
};

#endif




