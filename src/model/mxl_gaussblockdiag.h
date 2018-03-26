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
		int R; //number of draws


	public:
		MxlGaussianBlockDiag(CSR_matrix xf, std::vector<int> lbl,
				int numclass, int dim, bool zeroinit, 
				int numdraws);

		double negativeLogLik();

		void propensityFunction(int sampleID, 
			const std::vector<double> &normalDraws,
			std::vector<double> &classPropensity);

		void multinomialProb(const std::vector<double> &propensityScore, 
				std::vector<double> &mnProb);

		void simulatedProbability(int sampleID, const std::vector<double> &normalrv, std::vector<double> &simProb);

		void simulatedProbability_inline(int sampleID, std::vector<double> &simProb);


		void gradient(int sampleID, std::vector<double> &constantGrad, 
		ClassMeans &meanGrad, BlockCholeskey &covGrad);

		BlockCholeskey getCovCholeskey() {return this->covCholeskey;}


};

#endif
