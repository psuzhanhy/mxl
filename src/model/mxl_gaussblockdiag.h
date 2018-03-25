#ifndef MXL_GAUSSBLOCKDIAG_H
#define MXL_GAUSSBLOCKDIAG_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "param.h"
#include "matvec.h"
#include "mxl.h"

class MxlGaussianBlockDiag : public MixedLogit 
{
	
	private:
		ClassMeans means;
		BlockCholeskey covCholeskey;
		std::vector<double> classConstants;
		//boost::mt19937 rngseed; //seed for SAA
		int R; //number of draws

	public:
		MxlGaussianBlockDiag(CSR_matrix xf, std::vector<int> lbl,
							int numclass, int dim, bool zeroinit, 
							int numdraws);

		std::vector<double> propensityFunction(int sampleID, 
							std::vector<double> normalDraws);

		//TODO: implement pure virtual in .cpp 
		double negativeLogLik() { return 0.0; }
		std::vector<double> multinomialProb(std::vector<double> propensityScore);

		BlockCholeskey getCovCholeskey() 
		{
			return this->covCholeskey;  
		}


		
	
};

#endif
