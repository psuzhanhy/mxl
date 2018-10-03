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
        _beta(numclass, dim, zeroinit)
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
	if (zeroinit)
	{
		for(int k=0; k<this->numClass-1; k++)
			this->_intercept[k] = 0;
	} 
	else
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
	 * classProb is an array of K-1 containing the probability of each class 
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


void LogisticRegression::fit_by_SGD(double initStepSize, int maxEpochs, OptHistory &history)
{
	
}
