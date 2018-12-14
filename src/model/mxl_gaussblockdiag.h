#ifndef MXL_GAUSSBLOCKDIAG_H
#define MXL_GAUSSBLOCKDIAG_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "param.h"
#include "matvec.h"
#include "logistic.h"
#include "common.h"

#include "opthistory.h"


class MxlGaussianBlockDiag : public Logistic 
{
	
	private:
		ClassMeans means;
		BlockCholeskey covCholeskey;
		std::vector<double> classConstants;

		int R; //number of draws


	public:
		MxlGaussianBlockDiag(CSR_matrix xf, std::vector<int> lbl,
				int numclass, int dim, bool zeroinit,int numdraws);

		double negativeLogLik(const std::vector<double> &classConstants,
	    		const ClassMeans &means,
				const BlockCholeskey &covCholeskey,
				int numThreads);

		double negativeLogLik() 
		{
			return negativeLogLik(classConstants, means, 
					covCholeskey, CommonUtility::numSecondaryThreads);
		}

		double objValue(const std::vector<double> &classConstants,
				const ClassMeans &means,
				const BlockCholeskey &covCholeskey, int numThreads);

		double objValue()
		{
			return objValue(classConstants, means, 
					covCholeskey, CommonUtility::numSecondaryThreads);		
		}
		
		void propensityFunction(const std::vector<double> &classConstants,
	    		const ClassMeans &means, const BlockCholeskey &covCholeskey,
	 			int sampleID, const std::vector<double> &normalDraws,
				std::vector<double> &classPropensity);

		void multinomialProb(const std::vector<double> &propensityScore, 
				std::vector<double> &mnProb);

		void simulatedProbability(
				const std::vector<double> &classConstants,
				const ClassMeans &means,
				const BlockCholeskey &covCholeskey,
				int sampleID, 
				const std::vector<double> &normalrv, 
				std::vector<double> &simProb);

		void gradient(int sampleID, std::vector<double> &constantGrad,
				ClassMeans &meanGrad, BlockCholeskey &covGrad);

		void fit_by_SGD(double stepsize, std::string stepsizeRule, int maxEpochs, 
				OptHistory &history, bool writeHistory, bool adaptiveStop);

		void fit_by_APG(double stepsize, double momentum, 
				double momentumShrinkage, int maxIter,
				OptHistory &history, bool writeHistory);

		void fit_by_Hybrid(double stepsizeSGD, std::string stepsizeRule, 
				double stepsizeAGD, double momentum, double momentumShrinkage,
				int maxEpochs, OptHistory &history, bool writeHistory);
				
		double l2normsq(const ClassMeans &mean1,
				const BlockCholeskey &cov1,
				const std::vector<double> &constants1, 
				const ClassMeans &mean2,
				const BlockCholeskey &cov2,
				const std::vector<double> &constants2) const;

		double l2normsq(const ClassMeans &mean1,
				const BlockCholeskey &cov1,
				const std::vector<double> &constants1) const;


		double gradNormSq(ClassMeans &meanGrad,
				BlockCholeskey &covGrad,
				std::vector<double> &constantGrad,
				int numThreads);


		std::vector<int> insamplePrediction(const std::vector<double> &constants1,
				const ClassMeans &mean1,
				const BlockCholeskey &cov1);
		
		std::vector<int> insamplePrediction();

		BlockCholeskey getCovCholeskey() {return this->covCholeskey;}

		ClassMeans getMeans() {return this->means;}

		std::vector<double> getConstants() {return this->classConstants;}

};

#endif
