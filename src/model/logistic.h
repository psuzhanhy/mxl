#ifndef LOGISTIC_H
#define LOGISTIC_H 
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "matvec.h"


class Logistic 
{
	/*
	*  abstract base class for logistic models
	*/
	protected:
		int numSamples;
		int numClass;
		int dimension;
		CSR_matrix xfeature;
		std::vector<int> label;		

	public:
		Logistic(CSR_matrix xf, std::vector<int> lbl, int numclass, int dim): 
				xfeature(xf), label(lbl), numClass(numclass), dimension(dim) 
		{ numSamples = xf.number_rows; } 

		virtual double negativeLogLik() = 0;
		
		virtual ~Logistic() {}
		 	
};

#endif




