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
#include "mxl.h"
#include "mxl_gaussblockdiag.h"
#include "common.h"



MxlGaussianBlockDiag::MxlGaussianBlockDiag(CSR_matrix xf, std::vector<int> lbl,
		int numclass, int dim, bool zeroinit, int numdraws): 
		MixedLogit(xf,lbl,numclass,dim), 
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



void MxlGaussianBlockDiag::propensityFunction(int sampleID, 
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
		meanPropensity[k] = this->xfeature.rowInnerProduct(this->means.meanVectors[k],
					this->means.dimension, sampleID);

	for(int r=0; r<this->R; r++)
	{
		for(int k=0; k<this->numClass; k++)
		{
			int start = r * this->numClass * this->dimension + k * this->dimension;
			int end = start + this->dimension - 1;

			double randomPropensity = this->covCholeskey.factorArray[k].quadraticForm(this->xfeature, 
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



void MxlGaussianBlockDiag::simulatedProbability(int sampleID, 
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
	this->propensityFunction(sampleID, normalrv, propensityScore); 	
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



double MxlGaussianBlockDiag::negativeLogLik() 
{
	double nll = 0;
	#pragma omp parallel num_threads(CommonUtility::numThreads)
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
		this->simulatedProbability(n,normalrv, probSim);//populate probSim 
		int lbl = this->label[n];
		double probSAA = 0;
		for(int r=0; r<this->R; r++)
			probSAA += probSim[r*this->numClass+lbl];
		nll += log(probSAA/this->R); 
	} // pragma omp for reduction(+:nll)
	} //pragma omp parallel num_threads(2)

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
	this->simulatedProbability(sampleID,normalrv, probSim);//populate probSim 
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
			summationDraws[k*this->dimension+i] /= (probSAA * this->R);

        /* update gradient with respect to constant term */
        constantGrad[k] += summationTerm[k]/(probSAA * this->R);
		 
        /* update gradient with respect to mean */
        for(int i= this->xfeature.row_offset[sampleID]; i<xfeature.row_offset[sampleID+1];i++) 
            meanGrad.meanVectors[k][xfeature.col[i]] += summationTerm[k]/(probSAA * this->R)
 					* xfeature.val[i];
	
        /* update gradient with respect to Covariance Choleskey Factor */
		int start = k * this->dimension;
		int end = start + this->dimension - 1;
        this->xfeature.rowOuterProduct2LowerTri(sampleID, 
				summationDraws, start, end, 
				covGrad.factorArray[k]);

	}

} // MxlGaussianBlockDiag::gradient


void MxlGaussianBlockDiag::fit(double stepsize, double scalar, int maxEpochs)
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

	for(int t=0; t<maxEpochs; t++)
	{
		#pragma omp parallel num_threads(CommonUtility::numThreads) firstprivate(meanGrad,covGrad,constantGrad) 
		{
        int nThreads = omp_get_num_threads();	
		int tid = omp_get_thread_num();	
		std::vector<double> ordering(this->numSamples/nThreads);	
		for(int n=0; n<this->numSamples/nThreads; n++)
		{
			ctr[0] = t;
			ctr[1] = numSamples/nThreads * tid + n;
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)
			double x = r123::uneg11<double>(unidrand[0]);
			int r = 0.5 * (x + 1.0) * (this->numSamples-1);//discretized range
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
	}	
} //MxlGaussianBlockDiag::fit


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
