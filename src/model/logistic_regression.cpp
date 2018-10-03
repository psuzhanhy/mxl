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
#include "logistic_regression.h"
#include "common.h"
#include "opthistory.h"


LogisticRegression::LogisticRegression(CSR_matrix xf, std::vector<int> lbl, 
        int numclass, int dim, bool zeroinit): 
        Logistic(xf,lbl,numclass,dim),
        _beta(numclass, dim, zeroinit),
		_intercept(numClass-1, 0.0)
{
	/*	
	*	ctor for LogisticRegression class
	*/
	try
	{
		if (this->_beta.dimension != xf.number_cols)
			throw "mean dimension and feature dimension does not match.\n";

	} catch (const char* msg){
		std::cout << msg << std::endl;
	}

	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = static_cast<int> (time(nullptr));//seed	
	if (!zeroinit)
	{				
		for(int k=0; k<this->numClass-1; k++)	
		{
        	ctr[0] = k;	
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)	
			double x = r123::uneg11<double>(unidrand[0]);	
			this->_intercept[k] = 0.5*(x+1.0);
		}
	}

} // end of ctor


void LogisticRegression::multinomialProb(int sampleID, std::vector<double> &classProb, 
            Beta &beta, std::vector<double>& intercept)  
{
	/*
	 * compute multinomial probabilities
	 * classProb is an array of K containing the probability of each class 
	 */

    std::vector<double> linearPredictor(numClass,0.0);
    for(int k=0; k<numClass-1; k++)
        linearPredictor[k] = this->xfeature.rowInnerProduct(beta.beta[k], beta.dimension, sampleID)
                            + intercept[k];
    linearPredictor[numClass-1] = 0.0;

    double maxscore = linearPredictor[0];
    for(int i=0; i<this->numClass; i++) 
    {
        if (linearPredictor[i] > maxscore)
            maxscore = linearPredictor[i];
    }

    double sumProb = 0;
    for(int k=0; k<this->numClass; k++)
        sumProb += exp(linearPredictor[k]-maxscore);
   
    for(int k=0; k<this->numClass; k++)
        classProb[k] = exp(linearPredictor[k]-maxscore)/sumProb;    

}


double LogisticRegression::negativeLogLik(Beta& beta, std::vector<double> &intercept)
{
	/* return negative LogLikelihood */
	std::vector<double> 
	for(int n=0; n<this->numSamples; n++)
	{

	}
}


void LogisticRegression::fit_by_SGD(double initStepSize, int batchSize, 
		int maxIter, OptHistory &history)
{
	Beta betaTemp(this->_beta);
	Beta betaGrad(numClass, dimension, true);
	std::vector<double> interceptTemp(this->_intercept);
	std::vector<double> interceptGrad(numClass-1,0.0);
	double stepsize = initStepSize;
	// random generator
	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = CommonUtility::time_start_int;//seed		
	for(int t=0; t<maxIter; t++)
	{
		std::vector<double> classProb(numClass);	
		std::fill(interceptGrad.begin(), interceptGrad.end(), 0.0);
		betaGrad.setzero();
		// sample selection and batch gradient
		for(int b=0; b<batchSize; b++)
		{
			ctr[0] = t;
			ctr[1] = b;
			//unif(-1,1)
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); 
			double x = r123::uneg11<double>(unidrand[0]);
			//discretized range
			int r = 0.5 * (x + 1.0) * (this->numSamples-1);
			this->multinomialProb(r, classProb, betaTemp, interceptTemp);
			for(int k=0; k<numClass-1; k++)
			{
				double lbl = (label[r]==k ? 1.0 : 0.0);
				double coeff =  classProb[k] - lbl;
				interceptGrad[k] = coeff;
				for(int i=xfeature.row_offset[r]; i<xfeature.row_offset[r+1];i++) 
					betaGrad.beta[k][xfeature.col[i]] += coeff * xfeature.val[i];
			}
		}		
		// averaging and increment
		stepsize = initStepSize/(t+1);
		for(int k=0; k<numClass; k++)
		{
			interceptGrad[k] *= (1/batchSize);
			interceptTemp[k] -= stepsize * interceptGrad[k];
		}
		betaGrad *= (1/batchSize) * stepsize;
		betaTemp -= betaGrad;
	}	
	this->_beta = betaTemp;
	this->_intercept = interceptTemp;

} //end of fit_by_SGD
