#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <sys/time.h>
#include <omp.h>
#include <iostream>
#include <Random123/threefry.h>
#include <Random123/philox.h>
#include "uniform.hpp"
#include "boxmuller.hpp"
#include "matvec.h"
#include "param.h"
#include "logistic.h"
#include "mxl_gaussblockdiag.h"
#include "common.h"
#include "opthistory.h"


MxlGaussianBlockDiag::MxlGaussianBlockDiag(CSR_matrix xf, std::vector<int> lbl,
		int numclass, int dim, bool zeroinit, int numdraws): 
		Logistic(xf,lbl,numclass,dim), 
		means(numclass, dim, zeroinit),
		covCholeskey(numclass, dim, zeroinit),
		classConstants(numclass),
		R(numdraws)
{
	/*	
	*	ctor for MxlGaussianBlockDiag class
	*/
	try
	{
		if (this->means.dimension != xf.number_cols)
			throw "mean dimension and feature dimension does not match.\n";
		if (this->covCholeskey.dimension != xf.number_cols)
			throw "covariance dimension and feature dimension does not match.\n";
	} catch (const char* msg){
		std::cout << msg << std::endl;
	}

	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = static_cast<int> (time(nullptr));//seed	
	if (zeroinit)
	{
		for(int k=0; k<this->numClass; k++)
			this->classConstants[k] = 0;
	} 
	else
	{					
		for(int k=0; k<this->numClass; k++)	
		{
        	ctr[0] = k;	
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
			double x = r123::uneg11<double>(unidrand[0]);	
			this->classConstants[k] = 0.5*(x+1.0);
		}
	}
} 



void MxlGaussianBlockDiag::propensityFunction(const std::vector<double> &classConstants,
	    const ClassMeans &means,
		const BlockCholeskey &covCholeskey,
		int sampleID, 
		const std::vector<double> &normalDraws,
		std::vector<double> &classPropensity) 
{
	/*
	* compute propensity score for observation sampleID, and simulated 
	random normal draw
	
	@param: std::vector<double> normalDraws: vector of length numClass * dimension * R
	@return: std::vector<double> classPropensity: vector of length R * numClass
		group by each simulation [(c1,r1) .. (C,r1) | ... | (c1,R)...(C,R)]
	*/
	std::vector<double> meanPropensity(this->numClass); 
	for(int k=0; k<this->numClass; k++)
		meanPropensity[k] = this->xfeature.rowInnerProduct(means.meanVectors[k],
					means.dimension, sampleID);

	for(int r=0; r<this->R; r++)
	{
		for(int k=0; k<this->numClass; k++)
		{
			int start = r * this->numClass * this->dimension + k * this->dimension;
			int end = start + this->dimension - 1;

			double randomPropensity = covCholeskey.factorArray[k].quadraticForm(this->xfeature, 
					sampleID, normalDraws,start,end);

			classPropensity[r*this->numClass+k] = meanPropensity[k]+randomPropensity+classConstants[k];
		}
	}	
}


void MxlGaussianBlockDiag::multinomialProb(
		const std::vector<double> &propensityScore, 
		std::vector<double> &mnProb)
{
	/*
	 * compute multinomial probabilities
	 * classProb is an array containing the probability of each class 
	 */

    double maxscore = propensityScore[0];
    for(int i=0; i<this->numClass; i++) 
    {
        if (propensityScore[i] > maxscore)
            maxscore = propensityScore[i];
    }

    double sumProb = 0;
    for(int k=0; k<this->numClass; k++)
        sumProb += exp(propensityScore[k]-maxscore);
   
    for(int k=0; k<this->numClass; k++)
        mnProb[k] = exp(propensityScore[k]-maxscore)/sumProb;
}



void MxlGaussianBlockDiag::simulatedProbability(
		const std::vector<double> &classConstants,
	    const ClassMeans &means,
		const BlockCholeskey &covCholeskey,
		int sampleID, 
		const std::vector<double> &normalrv, 
		std::vector<double> &simProb)
{
	/*
	*	return a vector of length (R * numClass) of the 
	*	simulated probabilities for each class
	* 	@param:
	*		sampleID: computing the simulated probabilities of (R * numClass) for sampleID
	*		simProb:  empty vector, populated and return by the function
	*/

	// length R * numClass
	std::vector<double> propensityScore(this->numClass * this->R);
	this->propensityFunction(classConstants, means, covCholeskey,
			sampleID, normalrv, propensityScore); 	

 	for(int r=0; r<this->R; r++)
	{
		double maxscore = propensityScore[r*this->numClass];
		for(int k=0; k<this->numClass; k++)
		{
			if (propensityScore[r*this->numClass+k] > maxscore)  
				maxscore = propensityScore[r*this->numClass+k];
		}

	    double sumProb = 0;
	    for(int k=0; k<this->numClass; k++)
		{
			propensityScore[r*this->numClass+k] -= maxscore;
			propensityScore[r*this->numClass+k] = exp(propensityScore[r*this->numClass+k]);
	        sumProb += propensityScore[r*this->numClass+k];
	 	}	

		for(int k=0; k<this->numClass; k++)
			simProb[r*this->numClass+k] = propensityScore[r*this->numClass+k]/sumProb;	
	}
}



void MxlGaussianBlockDiag::simulatedProbability_inline(int sampleID, std::vector<double> &simProb)
{
/*** this method is deprecated ****/

	// generate normal(0,1) vector 
	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = CommonUtility::time_start_int;//seed	

	std::vector<double> meanPropensity(this->numClass); 
	for(int k=0; k<this->numClass; k++)
		meanPropensity[k] = this->xfeature.rowInnerProduct(this->means.meanVectors[k],
					this->means.dimension, sampleID);

	std::vector<double> propensityScore(this->numClass);
	int rvdim = this->numClass * this->dimension; 
	std::vector<double> normalrv(rvdim);
	ctr[0] = sampleID;
	for(int r=0; r<this->R; r++)
	{
		// random draws
		for(int i=0; i<rvdim/2; i++)
		{
			ctr[1] = i;
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
			auto nr = r123::boxmuller(unidrand[0], unidrand[1]);	
			normalrv[2*i] = nr.x;
			normalrv[2*i+1] = nr.y;	
		}
		ctr[1] = rvdim/2;
        CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
		auto nr = r123::boxmuller(unidrand[0], unidrand[1]);
		normalrv[rvdim-1] = nr.x;
	
		for(int k=0; k<this->numClass; k++)
		{
			int start = k * this->dimension;
			int end = start + this->dimension - 1;

			double randomPropensity = this->covCholeskey.factorArray[k].quadraticForm(this->xfeature, 
					sampleID, normalrv,start,end);

			propensityScore[k] = meanPropensity[k]+randomPropensity+classConstants[k];
		}
			
		double maxscore = propensityScore[0];
		for(int k=0; k<this->numClass; k++)
		{
			if (propensityScore[k] > maxscore)  
				maxscore = propensityScore[k];
		}

	    double sumProb = 0;
	    for(int k=0; k<this->numClass; k++)
		{
			propensityScore[k] -= maxscore;
			propensityScore[k] = exp(propensityScore[k]);
	        sumProb += propensityScore[k];
	 	}	

		for(int k=0; k<this->numClass; k++)
			simProb[r*this->numClass+k] = propensityScore[k]/sumProb;	
	}	
}// MxlGaussianBlockDiag::simulatedProbability_inline



double MxlGaussianBlockDiag::negativeLogLik(
		const std::vector<double> &classConstants,
	    const ClassMeans &means,
		const BlockCholeskey &covCholeskey, int numThreads) 
{
	double nll = 0;
	#pragma omp parallel num_threads(numThreads)
	{
	int rvdim = this->numClass * this->R * this->dimension;
	//normal(0,1) draws for sampleID
	std::vector<double> normalrv(rvdim);

	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = CommonUtility::time_start_int;//seed	
	#pragma omp for reduction(+:nll) 
	for(int n=0; n<this->numSamples; n++)
	{
		ctr[0] = n;		
		for(int i=0; i<rvdim/2; i++)
		{
			ctr[1] = i;
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
			auto nr = r123::boxmuller(unidrand[0], unidrand[1]);	
			normalrv[2*i] = nr.x;
			normalrv[2*i+1] = nr.y;			
		}
		ctr[1] = rvdim/2;
        CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
		auto nr = r123::boxmuller(unidrand[0], unidrand[1]);
		normalrv[rvdim-1] = nr.x;
	
		std::vector<double> probSim(this->R * this->numClass);
		//populate probSim 
		this->simulatedProbability(classConstants, means, 
				covCholeskey, n,normalrv, probSim);
 
		int lbl = this->label[n];
		double probSAA = 0;
		for(int r=0; r<this->R; r++)
			probSAA += probSim[r*this->numClass+lbl];
		nll += log(probSAA/this->R); 
	} // pragma omp for reduction(+:nll)
	} //pragma omp parallel num_threads

	return -nll;
}


void MxlGaussianBlockDiag::gradient(int sampleID, 
		std::vector<double> &constantGrad,
		ClassMeans &meanGrad, 
		BlockCholeskey &covGrad)
{
	// generate normal(0,1) vector
	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = CommonUtility::time_start_int;//seed	
	ctr[0] = sampleID;
	int rvdim = this->numClass * this->R * this->dimension; 
	std::vector<double> normalrv(rvdim);
	// all random draws for sampleID
	for(int i=0; i<rvdim/2; i++)
	{
		ctr[1] = i;
		CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
		auto nr = r123::boxmuller(unidrand[0], unidrand[1]);	
		normalrv[2*i] = nr.x;
		normalrv[2*i+1] = nr.y;			
	}
	ctr[1] = rvdim/2;
	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
	auto nr = r123::boxmuller(unidrand[0], unidrand[1]);
	normalrv[rvdim-1] = nr.x;

	std::vector<double> probSim(this->R * this->numClass);
	//populate probSim 
	this->simulatedProbability(this->classConstants, this->means, 
			this->covCholeskey, sampleID,normalrv, probSim);
	// Sample Average Approximated Probability
	int lbl = this->label[sampleID];
	double probSAA = 0;
	for(int r=0; r<this->R; r++)
		probSAA += probSim[r*this->numClass+lbl];
	probSAA /= this->R; 

	std::vector<double> summationTerm(this->numClass,0.0);	
	std::vector<double> summationDraws(this->numClass * this->dimension,0.0);
	
	for(int k=0; k<this->numClass; k++)
	{
		double b;
		for(int r=0 ; r<R ; r++)
		{
			if (k != lbl)
			{
				b = - probSim[r*this->numClass+k];
			} else {
				b = (1-probSim[r*this->numClass+k]);
			}	
			summationTerm[k] -= probSim[r*this->numClass+lbl] * b;
			for(int i=0 ; i<this->dimension ; i++)
			   summationDraws[k*this->dimension+i] -= probSim[r*this->numClass+lbl] * b 
					* normalrv[r*this->numClass*this->dimension+k*this->dimension+i];
		}
		
		for(int i=0; i<this->dimension; i++)
			summationDraws[k*this->dimension+i] /= (probSAA * this->R * this->numSamples);

        /* update gradient with respect to constant term */  
		double increment =  summationTerm[k]/(probSAA * this->R * this->numSamples);
		#pragma omp atomic
        constantGrad[k] += increment;		 
        /* update gradient with respect to mean */
        for(int i= this->xfeature.row_offset[sampleID]; i<xfeature.row_offset[sampleID+1];i++) 
		{
        	increment =  summationTerm[k]/(probSAA * this->R * this->numSamples) 
					* xfeature.val[i];
			#pragma omp atomic
			meanGrad.meanVectors[k][xfeature.col[i]] += increment;
		}
        /* update gradient with respect to Covariance Choleskey Factor */
		int start = k * this->dimension;
		int end = start + this->dimension - 1;
        this->xfeature.rowOuterProduct2LowerTri(sampleID, 
				summationDraws, start, end, 
				covGrad.factorArray[k]);

	}
	
} //MxlGaussianBlockDiag::gradient


void MxlGaussianBlockDiag::fit_by_SGD(double stepsize, double scalar, 
		int maxEpochs, OptHistory &history)
{
	/*
	*	fit Sample Average Approximated marginal log-likelihood with SGD 
	*/

	ClassMeans meanGrad(this->numClass, this->dimension, true);
	BlockCholeskey covGrad(this->numClass, this->dimension, true);
	std::vector<double> constantGrad(this->numClass);

	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = CommonUtility::time_start_int;//seed	

	history.nll[0] = this->negativeLogLik(); 
	std::cout << "intial NLL " << history.nll[0] << std::endl;	
	history.gradNormSq[0] = this->gradNormSq(meanGrad, covGrad,
			constantGrad, CommonUtility::numSecondaryThreads);
	std::cout << "Initial squared l2-norm of gradient: " 
			<< history.gradNormSq[0] << std::endl;	
	struct timeval start, finish;
	/***************************************************************************/
	/********************* start SGD *******************************************/	
	for(int t=0; t<maxEpochs; t++)
	{
		gettimeofday(&start,nullptr) ; // set timer start 
		ClassMeans meanOld(this->means);
		BlockCholeskey covOld(this->covCholeskey);
		std::vector<double> constantsOld(this->classConstants);
		#pragma omp parallel num_threads(CommonUtility::numThreads)\
		firstprivate(meanGrad,covGrad,constantGrad) 
		{
        int nThreads = omp_get_num_threads();	
		int tid = omp_get_thread_num();	
		std::vector<double> ordering(this->numSamples/nThreads);	
		for(int n=0; n<this->numSamples/nThreads; n++)
		{
			ctr[0] = t;
			ctr[1] = numSamples/nThreads * tid + n;
			//unif(-1,1)
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); 
			double x = r123::uneg11<double>(unidrand[0]);
			//discretized range
			int r = 0.5 * (x + 1.0) * (this->numSamples-1);
			ordering[n] = r;
		}
		stepsize *= scalar;
		for(int n=0; n<this->numSamples/nThreads; n++)
		{
			std::fill(constantGrad.begin(), constantGrad.end(), 0.0);
			covGrad.setzero();
			meanGrad.setzero();

			this->gradient(ordering[n],constantGrad, meanGrad, covGrad);
			// update mean
			meanGrad *= stepsize;
			this->means -= meanGrad;
			// update covariance
			covGrad *= stepsize;
			this->covCholeskey -= covGrad;
			// update constants
			for(int k=0; k<this->numClass; k++)
				classConstants[k] -= stepsize * constantGrad[k];		
		}	
		} //pragma omp parallel
		
		/***********************************************************************/	
		/***************** record optimization progress ************************/
    	gettimeofday(&finish,nullptr) ; // set timer finish      
		history.iterTime[t+1] = finish.tv_sec - start.tv_sec;     
    	history.iterTime[t+1] += (finish.tv_usec-start.tv_usec)/(1000.0 * 1000.0);   
		history.iterTime[t+1] += history.iterTime[t];

		history.gradNormSq[t+1] = this->gradNormSq(meanGrad, covGrad,
				constantGrad, CommonUtility::numSecondaryThreads);
		std::cout << "squared l2-norm of gradient after " << t+1 
				<<" iterations of SGD: " << history.gradNormSq[t+1] << std::endl;
	
		double change = this->l2normsq(this->means, this->covCholeskey, 
				this->classConstants, meanOld, covOld, constantsOld);
		history.paramChange[t+1] = sqrt(change);
		std::cout << "change of parameters in l2-norm in " << t+1 << " iteration: "
				<< history.paramChange[t+1] << std::endl;		
	
		history.nll[t+1] = this->negativeLogLik();
		std::cout << "NLL after " << t+1 << " iterations of SGD: " 
				<< history.nll[t+1] << std::endl;

		//TODO
		std::cout << this->testRay(this->classConstants, this->means, 
				this->covCholeskey,10, CommonUtility::numThreads) << std::endl;

		double paramNorm = this->l2normsq(this->means, this->covCholeskey, 
				this->classConstants);
		std::cout << paramNorm << std::endl;
		/***********************************************************************/	
	}	
} //MxlGaussianBlockDiag::fit_by_SGD


void MxlGaussianBlockDiag::fit_by_APG(double stepsize, double momentum, 
		double momentumShrinkage, int maxIter, OptHistory &history)
{

	ClassMeans meanGrad(this->numClass, this->dimension, true);
	BlockCholeskey covGrad(this->numClass, this->dimension, true);
	std::vector<double> constantGrad(this->numClass);
	// x iterates
	ClassMeans mean_x(this->means);
	BlockCholeskey cov_x(this->covCholeskey);
	std::vector<double> classConstants_x(this->classConstants);
	// x iterates new
	ClassMeans mean_x_new(this->means);
	BlockCholeskey cov_x_new(this->covCholeskey);
	std::vector<double> classConstants_x_new(this->classConstants);
	// v iterates
	ClassMeans mean_v(this->means);
	BlockCholeskey cov_v(this->covCholeskey);
	std::vector<double> classConstants_v(this->classConstants);

	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = CommonUtility::time_start_int;//seed	

	history.nll[0] = this->negativeLogLik(); 
	std::cout << "intial NLL " << history.nll[0] << std::endl;	
	struct timeval start, finish;
	/***************************************************************************/
	/********************* start APG *******************************************/	
	for(int t=0; t<maxIter; t++)
	{
		gettimeofday(&start,nullptr) ; // set timer start 
		std::fill(constantGrad.begin(), constantGrad.end(), 0.0);
		covGrad.setzero();
		meanGrad.setzero();
		#pragma omp parallel for num_threads(CommonUtility::numThreads) 
		for(int n=0; n<this->numSamples; n++)
			this->gradient(n,constantGrad, meanGrad, covGrad);
    	gettimeofday(&finish,nullptr) ; // pause timer 
		history.iterTime[t+1] = finish.tv_sec - start.tv_sec;     
    	history.iterTime[t+1] += (finish.tv_usec-start.tv_usec)/(1000.0 * 1000.0); 
		//accounting for size of gradient
		history.gradNormSq[t] = this->l2normsq(meanGrad,covGrad,constantGrad);
		std::cout << "squared l2-norm of gradient after " << t
				<<" iterations of APG: " << history.gradNormSq[t] << std::endl;
		gettimeofday(&start,nullptr) ; // resume timer

		// update x iterates
		classConstants_x = classConstants_x_new;		
		mean_x = mean_x_new;
		cov_x = cov_x_new;
		for(int k=0; k<this->numClass; k++)
			classConstants_x_new[k] = this->classConstants[k]-stepsize*constantGrad[k];
		meanGrad *= stepsize;
		mean_x_new = this->means-meanGrad;
		covGrad *= stepsize;
		cov_x_new = this->covCholeskey-covGrad;
		// update v iterates
		for(int k=0; k<this->numClass; k++)
			classConstants_v[k] = classConstants_x_new[k] 
					+ (classConstants_x_new[k]-classConstants_x[k]) * momentum;
		mean_v = mean_x_new + (mean_x_new - mean_x) * momentum;
		cov_v = cov_x_new + (cov_x_new - cov_x) * momentum;
		//compare NLL
		double nll_x = this->negativeLogLik(classConstants_x_new, mean_x_new, 
				cov_x_new, CommonUtility::numThreads);	
		double nll_v = this->negativeLogLik(classConstants_v, mean_v, 
				cov_v, CommonUtility::numThreads);
    	gettimeofday(&finish,nullptr) ; // set timer finish      
		history.iterTime[t+1] += finish.tv_sec - start.tv_sec;     
    	history.iterTime[t+1] += (finish.tv_usec-start.tv_usec)/(1000.0 * 1000.0);   
		history.iterTime[t+1] += history.iterTime[t];

		/***********************************************************************/	
		/***** choosing iteraties and record optimization progress *************/
		if (nll_x <= nll_v)
		{
			history.nll[t+1] = nll_x; 
			std::cout << "NLL after " << t+1 << " iterations of APG: "
					<< history.nll[t+1] << std::endl;	
			double change = l2normsq(this->means, this->covCholeskey, 
					this->classConstants, mean_x_new, cov_x_new, classConstants_x_new);	
			history.paramChange[t+1] = sqrt(change);						
			std::cout << "change of parameters in l2-norm in " << t+1 << " iteration: "
					<< history.paramChange[t+1] << std::endl;	
		
			this->classConstants = classConstants_x_new;
			this->means = mean_x_new;
			this->covCholeskey = cov_x_new;
			momentum *= momentumShrinkage;

		} else 
		{
			history.nll[t+1] = nll_v; 
			std::cout << "NLL after " << t+1 << " iterations of APG: "
					<< history.nll[t+1] << std::endl;	
			double change = l2normsq(this->means, this->covCholeskey, 
					this->classConstants, mean_v, cov_v, classConstants_v);	
			history.paramChange[t+1] = sqrt(change);	
			std::cout << "change in l2-norm in " << t+1 << " iteration: "
					<< history.paramChange[t+1] << std::endl;	

			this->classConstants = classConstants_v;
			this->means = mean_v;
			this->covCholeskey = cov_v;
			momentum = (momentum/(momentumShrinkage) < 1 ? momentum/(momentumShrinkage): 1);
		}
		/***********************************************************************/
		//TODO
		std::cout << this->testRay(this->classConstants, this->means, 
				this->covCholeskey,10, CommonUtility::numThreads) << std::endl;

		double paramNorm = this->l2normsq(this->means, this->covCholeskey, 
				this->classConstants);
		std::cout << paramNorm << std::endl;
	
	}
	
	history.gradNormSq[maxIter] = this->gradNormSq(meanGrad, covGrad,
			constantGrad, CommonUtility::numSecondaryThreads);
	std::cout << "squared l2-norm of gradient after " << maxIter
			<<" iterations of APG: " << history.gradNormSq[maxIter] << std::endl;

} //fit_by_APG


double MxlGaussianBlockDiag::l2normsq(const ClassMeans &mean1,
		const BlockCholeskey &cov1,
		const std::vector<double> &constants1, 
		const ClassMeans &mean2,
		const BlockCholeskey &cov2,
		const std::vector<double> &constants2) const
{
	
	double norm = 0.0;
	for(int k=0; k<this->numClass; k++)	
		norm += (constants1[k] - constants2[k]) * (constants1[k] - constants2[k]);

	BlockCholeskey covDiff = cov1 - cov2;
	ClassMeans meanDiff = mean1 - mean2;

	norm += covDiff.l2normsq();
	norm += meanDiff.l2normsq();

	return norm;
}

double MxlGaussianBlockDiag::l2normsq(const ClassMeans &mean1,
									const BlockCholeskey &cov1,
									const std::vector<double> &constants1) const
{	
	double norm = 0.0;
	for(int k=0; k<this->numClass; k++)	
		norm += constants1[k] * constants1[k];

	norm += cov1.l2normsq();
	norm += mean1.l2normsq();

	return norm;
}


double MxlGaussianBlockDiag::gradNormSq(ClassMeans &meanGrad,
										BlockCholeskey &covGrad,
										std::vector<double> &constantGrad,
										int numThreads) 
{
	std::fill(constantGrad.begin(), constantGrad.end(), 0.0);
	covGrad.setzero();
	meanGrad.setzero();
	#pragma omp parallel for num_threads(numThreads) 
	for(int n=0; n<this->numSamples; n++)
		this->gradient(n,constantGrad, meanGrad, covGrad);
	
	double gradnorm = this->l2normsq(meanGrad,covGrad,constantGrad);
	return gradnorm;
}


std::vector<int> MxlGaussianBlockDiag::insamplePrediction(const std::vector<double> &constants1,
		const ClassMeans &mean1,
		const BlockCholeskey &cov1)
{

	std::vector<int> fitlabel(this->numSamples);
	#pragma omp parallel num_threads(CommonUtility::numSecondaryThreads)
	{
	int rvdim = this->numClass * this->R * this->dimension;
	//normal(0,1) draws for sampleID
	std::vector<double> normalrv(rvdim);

	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = CommonUtility::time_start_int;//seed	
	#pragma omp for  
	for(int n=0; n<this->numSamples; n++)
	{
		ctr[0] = n;		
		for(int i=0; i<rvdim/2; i++)
		{
			ctr[1] = i;
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
			auto nr = r123::boxmuller(unidrand[0], unidrand[1]);	
			normalrv[2*i] = nr.x;
			normalrv[2*i+1] = nr.y;			
		}
		ctr[1] = rvdim/2;
        CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
		auto nr = r123::boxmuller(unidrand[0], unidrand[1]);
		normalrv[rvdim-1] = nr.x;
	
		std::vector<double> probSim(this->R * this->numClass);
		//populate probSim 
		this->simulatedProbability(constants1, mean1, 
				cov1, n,normalrv, probSim);
 
		std::vector<double> probSAA(this->numClass,0.0);
		for(int r=0; r<this->R; r++)
		{
			for(int k=0; k<this->numClass; k++)
				probSAA[k] += probSim[r*this->numClass+k];
		}
		int maxclass = this->label[n];
		double maxprob = probSAA[maxclass];
		for(int k=0; k<this->numClass; k++)
		{
			if (probSAA[k]>maxprob)
			{
				maxclass = k;
				maxprob = probSAA[k];
			}
		}
		fitlabel[n] = maxclass;	
	} // pragma omp for 
	} //pragma omp parallel num_threads
	return fitlabel;
} //MxlGaussianBlockDiag::insamplePrediction


std::vector<int> MxlGaussianBlockDiag::insamplePrediction() 
{
	return this->insamplePrediction(this->classConstants, this->means,this->covCholeskey);
}


double MxlGaussianBlockDiag::testRay(
		const std::vector<double> &classConstants,
	    const ClassMeans &means,
		const BlockCholeskey &covCholeskey, 
		int scalar,int numThreads) 
{
	std::vector<double> constants1(this->numClass);
	for(int k=0; k<this->numClass; k++)
		constants1[k] = scalar * classConstants[k];
	ClassMeans means1(means);
	BlockCholeskey cov1(covCholeskey);
	means1 *= scalar;
	cov1 *= scalar;
	double nll =this->negativeLogLik(constants1, means1, cov1, numThreads);
	return nll;
}

