#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <sys/time.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/generator_iterator.hpp>
#include "matvec.h"
#include "param.h"
#include "mxl.h"
#include "mxl_gaussblockdiag.h"
#include "rnggenerator.h"

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
		
	if (zeroinit)
	{
		for(int k=0; k<this->numClass; k++)
			this->classConstants[k] = 0;
	} 
	else
	{				
		for(int k=0; k<this->numClass; k++)				
			this->classConstants[k] = RngGenerator::unid_init();
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

	// generate normal(0,1) vector 
	RngGenerator::var_nor.engine().seed(sampleID);
	RngGenerator::var_nor.distribution().reset(); 

	std::vector<double> meanPropensity(this->numClass); 
	for(int k=0; k<this->numClass; k++)
		meanPropensity[k] = this->xfeature.rowInnerProduct(this->means.meanVectors[k],
					this->means.dimension, sampleID);

	std::vector<double> propensityScore(this->numClass);
	int rvdim = this->numClass * this->dimension; 
	std::vector<double> normalrv(rvdim);
	for(int r=0; r<this->R; r++)
	{
		// random draws
		for(int i=0; i<rvdim; i++)
			normalrv[i] = RngGenerator::var_nor();	

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
	int rvdim = this->numClass * this->R * this->dimension; 
	std::vector<double> normalrv(rvdim);

	for(int n=0; n<this->numSamples; n++)
	{
		// normal(0,1) draws for sampleID
		RngGenerator::var_nor.engine().seed(n);
		RngGenerator::var_nor.distribution().reset(); 
		for(int i=0; i<rvdim; i++)
			normalrv[i] = RngGenerator::var_nor();	

		std::vector<double> probSim(this->R * this->numClass); 
		this->simulatedProbability(n,normalrv, probSim);//populate probSim 

		int lbl = this->label[n];
		double probSAA = 0;
		for(int r=0; r<this->R; r++)
			probSAA += probSim[r*this->numClass+lbl];
		nll += log(probSAA/this->R); 
	}
	return -nll;
}


void MxlGaussianBlockDiag::gradient(int sampleID, 
		std::vector<double> &constantGrad,
		ClassMeans &meanGrad, 
		BlockCholeskey &covGrad)
{
	// generate normal(0,1) vector 
	RngGenerator::var_nor.engine().seed(sampleID);
	RngGenerator::var_nor.distribution().reset(); 
	int rvdim = this->numClass * this->R * this->dimension; 
	std::vector<double> normalrv(rvdim);
	// all random draws for sampleID
	for(int i=0; i<rvdim; i++)
		normalrv[i] = RngGenerator::var_nor();	

	std::vector<double> probSim(this->R * this->numClass); 
	this->simulatedProbability(sampleID,normalrv, probSim);//populate probSim 
	// Sample Average Approximated Probability
	double probSAA = 0;
	for(auto val : probSim)		
		probSAA += val;
	probSAA /= this->R; 

	std::vector<double> summationTerm(this->numClass,0.0);	
	std::vector<double> summationDraws(this->numClass * this->dimension,0.0);
	
	int lbl = this->label[sampleID];
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
        constantGrad[k] += summationTerm[k] / (probSAA * this->R); 
 
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

	boost::uniform_int<> unid_discrete(0, this->numSamples-1);
    boost::variate_generator<boost::mt19937&,boost::uniform_int<> > 
			unid_sampler(RngGenerator::rng, unid_discrete);

	ClassMeans meanGrad(this->numClass, this->dimension, true);
	BlockCholeskey covGrad(this->numClass, this->dimension, true);
	std::vector<double> constantGrad(this->numClass);
	std::vector<double> ordering(this->numSamples);
	
	for(int t=0; t<maxEpochs; t++)
	{
		for(int n=0; n<this->numSamples; n++)
			ordering[n] = unid_sampler();
		
		stepsize *= scalar;
		for(int n=0; n<this->numSamples; n++)
		{
			std::fill(constantGrad.begin(), constantGrad.end(), 0.0);
			covGrad.setzero();
			meanGrad.setzero();

			this->gradient(n,constantGrad, meanGrad, covGrad);
			// update mean
			meanGrad *= stepsize;
			this->means += meanGrad;
			// update covariance
			covGrad *= stepsize;
			this->covCholeskey += covGrad;
			// update constants
			for(int k=0; k<this->numClass; k++)
				classConstants[k] += stepsize * constantGrad[k];					
		}	
	}	
}


