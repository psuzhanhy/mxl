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
#include <math.h>

LogisticRegression::LogisticRegression(CSR_matrix xf, std::vector<int> lbl, 
        int numclass, int dim, double l1Lambda, double l2Lambda, bool zeroinit): 
        Logistic(xf,lbl,numclass,dim),
        _beta(numclass, dim, zeroinit),
		_intercept(numClass-1, 0.0),
		_l1Lambda(l1Lambda), _l2Lambda(l2Lambda)
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


void LogisticRegression::setL1Lambda(double l1Lambda)
{
	this->_l1Lambda = l1Lambda;
}


Beta LogisticRegression::getBeta()
{
	Beta betaCopy(this->_beta);
	return betaCopy;
} 


std::vector<double> LogisticRegression::getIntercept()
{
	std::vector<double> interceptCopy(this->_intercept);
	return interceptCopy;
}


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
	std::vector<double> classProb(numClass);
	double nll = 0.0;
	for(int n=0; n<this->numSamples; n++)
	{
		this->multinomialProb(n, classProb, beta, intercept);
		nll -= log(classProb[this->label[n]]);
	}
	return nll;
}


double LogisticRegression::negativeLogLik() 
{
	return negativeLogLik(this->_beta, this->_intercept);
}


double LogisticRegression::l1Regularizer(Beta& beta) const
{
	double penaltyval = 0.0;
	for(int k=0; k<this->numClass-1; k++)
	{
		for(int i=0; i<dimension; i++)
			penaltyval += this->_l1Lambda * fabs(beta.beta[k][i]);
	}
	return penaltyval;
}


double LogisticRegression::l2Regularizer(Beta& beta) const
{
	double penaltyval = 0.0;
	for(int k=0; k<this->numClass-1; k++)
	{
		for(int i=0; i<dimension; i++)
			penaltyval += this->_l2Lambda * beta.beta[k][i] * beta.beta[k][i];
	}
	return penaltyval;
}


double LogisticRegression::objValue(Beta& beta, std::vector<double> &intercept)
{
	/* 
	return obj vaue loss = 1/n * sum loss(x_i,y_i; beta, intercept)  + l1lambda * R(beta) 
	*/
	double fobjval = negativeLogLik(beta, intercept)/this->numSamples;
	if(this->_l1Lambda > 0.0)
		fobjval += l1Regularizer(beta);

	if(this->_l2Lambda > 0.0)
		fobjval += l2Regularizer(beta);

	return fobjval;
}


void LogisticRegression::stochasticGradient(int sampleID, 
		Beta& beta, std::vector<double> &intercept,
		Beta& betaGrad, std::vector<double> &interceptGrad)  
{
	/* compute stochastic gradient of loss using sampleID w.r.t beta and intercept */
	std::vector<double> classProb(numClass);	
	this->multinomialProb(sampleID, classProb, beta, intercept);
	for(int k=0; k<numClass-1; k++)
	{
		double lbl = (label[sampleID]==k ? 1.0 : 0.0);
		double coeff =  classProb[k] - lbl;
		interceptGrad[k] += coeff;
		for(int i=xfeature.row_offset[sampleID]; i<xfeature.row_offset[sampleID+1];i++) 
			betaGrad.beta[k][xfeature.col[i]] += coeff * xfeature.val[i];
	}
}


void LogisticRegression::l1Regularizer_Subgradient(Beta& beta, Beta& betaGrad) const
{
	/* write the subgradient w.r.t L1 regularizer on betaGrad */
	for(int k=0; k<this->numClass-1; k++)
	{
		for(int i=0; i<dimension; i++)
		{
			if(beta.beta[k][i]>0.0)
			{
				betaGrad.beta[k][i] += this->_l1Lambda;
			} else if(beta.beta[k][i]<0.0)
			{
				betaGrad.beta[k][i] -= this->_l1Lambda;
			}
		}
	}	
}


void LogisticRegression::l2Regularizer_Gradient(Beta& beta, Beta& betaGrad) const
{
	/* write the subgradient w.r.t L1 regularizer on betaGrad */
	for(int k=0; k<this->numClass-1; k++)
	{
		for(int i=0; i<dimension; i++)
			betaGrad.beta[k][i] += 2 * this->_l2Lambda * beta.beta[k][i];
	}	
}


void LogisticRegression::proximalL1(Beta& beta)
{
	/* 
	take beta as input of Proxy(beta) and modified beta 
	with proximal operator 
	*/
	for(int k=0; k<numClass-1; k++)
	{
		for(int i=0; i<dimension; i++)
		{
			if(fabs(beta.beta[k][i])<=this->_l1Lambda)
			{
				beta.beta[k][i]=0.0;
			} else {
				double softthresholding = fabs(beta.beta[k][i])-this->_l1Lambda; 
				beta.beta[k][i] = (beta.beta[k][i] > 0.0 ? 
						softthresholding : -1*softthresholding);
			}
		}
	}
}


void LogisticRegression::proximalSGD(double initStepSize, std::string stepsizeRule, int batchSize, 
		int maxIter, OptHistory &history, bool writeHistory, bool adaptiveStop)
{
	/* 
		solve  min 1/n * sum L(x_i,y_i, beta, intercept) + l1lambda * R(beta) 
	using Stochastic Proximal Gradient Descent 
	*/	
	// working variables
	Beta betaTemp(this->_beta);
	Beta betaGrad(numClass, dimension, true);
	Beta betaGradOld(numClass, dimension, true);

	std::vector<double> interceptTemp(this->_intercept);
	std::vector<double> interceptGrad(numClass-1,0.0);
	std::vector<double> interceptGradOld(numClass-1,0.0);

	// opt initialization 
	if (writeHistory)
	{
		history.iterTime.push_back(0.0);
		history.fobj.push_back(this->objValue(betaTemp, interceptTemp));
		for(int n=0; n<numSamples; n++)
			stochasticGradient(n, betaTemp, interceptTemp, betaGrad, interceptGrad);
		betaGrad *= 1/this->numSamples;
		l2Regularizer_Gradient(betaTemp, betaGrad); // L2
		l1Regularizer_Subgradient(betaTemp, betaGrad); // L1
		history.gradNormSq.push_back(betaGrad.l2normsq());
		for(int k=0; k<numClass-1; k++)
		{
			interceptGrad[k] /= this->numSamples;
			history.gradNormSq.back()  += interceptGrad[k]*interceptGrad[k];
		}
	}	
	double stepsize = initStepSize;	
	// random generator
	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = CommonUtility::time_start_int;//seed	
	int countDataPass = 0;
	int effectivePass = 0;
	struct timeval start, finish;
	// timer
	gettimeofday(&start,nullptr) ; // set timer start 
	int stoppingCounter = 0;
	for(int t=0; t<maxIter; t++)
	{
		//std::cout << "***************** iteration " << t << " *****************\n";
		std::fill(interceptGrad.begin(), interceptGrad.end(), 0.0);
		betaGrad.setzero();
		// sample selection and batch gradient
		for(int b=0; b<batchSize; b++)
		{
			ctr[0] = t+b;
			ctr[1] = b;
			// unif(-1,1)
        	CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); 
			double x = r123::uneg11<double>(unidrand[0]);
			//discretized range
			int r = 0.5 * (x + 1.0) * (this->numSamples-1);
			stochasticGradient(r, betaTemp, interceptTemp, betaGrad, interceptGrad);
		}		
		countDataPass += batchSize;
		
		for(int k=0; k<numClass-1; k++)
		{
			interceptGrad[k] *= (1.0/batchSize);
			interceptTemp[k] -= stepsize * interceptGrad[k];
		}
		betaGrad *= (1.0/batchSize);
		l2Regularizer_Gradient(betaTemp, betaGrad); //add gradient wrt L2 term
		betaTemp -= betaGrad * stepsize;
		if (stepsizeRule == "iterdecreasing")
			stepsize = initStepSize/(t+1);
	
		// proximal operator 
		proximalL1(betaTemp);
		if(countDataPass/this->numSamples>effectivePass)
		{
			effectivePass++;
			if (stepsizeRule == "epochdecreasing")
				stepsize = initStepSize/(effectivePass+1);

			if (writeHistory)
			{
				// timer
				gettimeofday(&finish,nullptr) ; // set timer start 
				history.iterTime.push_back(history.iterTime.back()+finish.tv_sec-start.tv_sec);     
				history.iterTime.back() += (finish.tv_usec-start.tv_usec)/(1000.0 * 1000.0); 
				// obj value
				history.fobj.push_back(this->objValue(betaTemp, interceptTemp));
				// first order optimality
				std::fill(interceptGrad.begin(), interceptGrad.end(), 0.0);
				betaGrad.setzero();			
				for(int n=0; n<numSamples; n++)
					stochasticGradient(n, betaTemp, interceptTemp, betaGrad, interceptGrad);
				betaGrad *= 1/this->numSamples;

				l2Regularizer_Gradient(betaTemp, betaGrad); // L2 
				l1Regularizer_Subgradient(betaTemp, betaGrad);// L1
				history.gradNormSq.push_back(betaGrad.l2normsq());
				for(int k=0; k<numClass-1; k++)
				{
					interceptGrad[k] /= this->numSamples;
					history.gradNormSq.back() += interceptGrad[k]*interceptGrad[k];
				}
				// reset timer
				gettimeofday(&start,nullptr) ; // set timer start 
			}

			if(adaptiveStop && writeHistory && history.fobj.back() > *(history.fobj.rbegin()+1))
				stoppingCounter++;	

			if(stepsizeRule == "funcvaladapt" && writeHistory && history.fobj.back() > *(history.fobj.rbegin()+1))
				stepsize *= 0.5;
			
		}

		if(stoppingCounter>=10)
		{
			std::cerr << "function value stopping condition met after " <<  effectivePass << " epochs, exit SGD\n";
			break;
		}

		if (history.gradNormSq.back() <= pow(10,-11))
		{
			std::cout << "first order optimality reached; stop opt process\n";
			break;
		}	
	}

	this->_beta = betaTemp;
	this->_intercept = interceptTemp;

} //end of proximalSGD


void LogisticRegression::hybridFirstOrder(double initSGDStepSize, int batchSizeSGD, std::string stepsizeRuleSGD, double stepSizeAGD,
		int maxEpochs, OptHistory &history)
{
        int maxIter = maxEpochs * numSamples / batchSizeSGD;
		this->proximalSGD(initSGDStepSize, stepsizeRuleSGD, batchSizeSGD,  
			maxIter, history, true, true);
		int epochsRunSGD = history.fobj.size()-1;  
		int epochsRunAGD = maxEpochs-epochsRunSGD;
		if (epochsRunAGD>0)
		{
			OptHistory historyAGD(epochsRunAGD);
			this->proximalAGD(initSGDStepSize, epochsRunAGD, historyAGD, true);	
			for(int t=1; t<historyAGD.fobj.size(); t++)
				history.fobj.push_back(historyAGD.fobj[t]);
			for(int t=1; t<historyAGD.gradNormSq.size(); t++)
				history.gradNormSq.push_back(historyAGD.gradNormSq[t]);
			for(int t=1; t<historyAGD.iterTime.size(); t++)
				history.iterTime.push_back(history.iterTime.back()+historyAGD.iterTime[t]);
		}
}



void LogisticRegression::proximalHybridBatchingGD(double stepSize, 
		int maxIter, OptHistory &history, bool writeHistory)
{
	/* 
		solve  min 1/n * sum L(x_i,y_i, beta, intercept) + l1lambda * R(beta) 
	using Stochastic Proximal Gradient Descent 
	*/	
	// working variables
	Beta betaTemp(this->_beta);
	Beta betaGrad(numClass, dimension, true);
	std::vector<double> interceptTemp(this->_intercept);
	std::vector<double> interceptGrad(numClass-1,0.0);
	// opt initialization 
	if (writeHistory)
	{
		history.iterTime.push_back(0.0);
		history.fobj.push_back(this->objValue(betaTemp, interceptTemp));
		for(int n=0; n<numSamples; n++)
			stochasticGradient(n, betaTemp, interceptTemp, betaGrad, interceptGrad);
		
		betaGrad *= 1/this->numSamples;
		l2Regularizer_Gradient(betaTemp, betaGrad); // L2
		l1Regularizer_Subgradient(betaTemp, betaGrad); // L1
		history.gradNormSq.push_back(betaGrad.l2normsq());
		for(int k=0; k<numClass-1; k++)
		{
			interceptGrad[k] /= this->numSamples;
			history.gradNormSq.back()  += interceptGrad[k]*interceptGrad[k];
		}
	}	

	int batchSize = 1;	
	int batchStart = 0;
	int countDataPass = 0;
	int effectivePass = 0;
	struct timeval start, finish;
	gettimeofday(&start,nullptr) ; // set timer start 
	for(int t=0; t<maxIter; t++)
	{
		//std::cout << "***************** iteration " << t << " *****************\n";
		//std::cout << "batch size: " << batchSize << std::endl;
		std::fill(interceptGrad.begin(), interceptGrad.end(), 0.0);
		betaGrad.setzero();
		// sample selection and batch gradient
		batchSize = (this->numSamples < 1.1*batchSize+1 ? this->numSamples : 1.1*batchSize+1);
		countDataPass += batchSize;
		if (batchSize < this->numSamples)
		{
			if(batchStart+batchSize-1 < this->numSamples)
			{	
				for(int b=batchStart; b<batchStart+batchSize; b++)
					stochasticGradient(b, betaTemp, interceptTemp, betaGrad, interceptGrad);
				batchStart = (batchSize+batchSize) % numSamples;			
			} 
			else 
			{
				for(int b=batchStart; b<numSamples; b++)
					stochasticGradient(b, betaTemp, interceptTemp, betaGrad, interceptGrad);
				batchStart = (batchSize+batchSize) % numSamples;			
				for(int b=0; b<batchStart; b++)
					stochasticGradient(b, betaTemp, interceptTemp, betaGrad, interceptGrad);
			}
		} 
		else
		{
			for(int b=0; b<batchSize; b++)
				stochasticGradient(b, betaTemp, interceptTemp, betaGrad, interceptGrad);
		}

		// averaging and increment
		for(int k=0; k<numClass-1; k++)
		{
			interceptGrad[k] *= (1.0/batchSize);
			interceptTemp[k] -= stepSize * interceptGrad[k];
		}
		betaGrad *= (1.0/batchSize);
		l2Regularizer_Gradient(betaTemp, betaGrad); //add gradient wrt L2 term
		betaTemp -= betaGrad * stepSize;

		// proximal operator 
		proximalL1(betaTemp);

		if (writeHistory & countDataPass/this->numSamples>effectivePass)
		{
			effectivePass++;
			// timer
			gettimeofday(&finish,nullptr) ; // set timer start 
			history.iterTime.push_back(history.iterTime.back()+finish.tv_sec-start.tv_sec);     
			history.iterTime.back() += (finish.tv_usec-start.tv_usec)/(1000.0 * 1000.0); 
			// obj value
			history.fobj.push_back(this->objValue(betaTemp, interceptTemp));
			// first order optimality
			std::fill(interceptGrad.begin(), interceptGrad.end(), 0.0);
			betaGrad.setzero();			
			for(int n=0; n<numSamples; n++)
				stochasticGradient(n, betaTemp, interceptTemp, betaGrad, interceptGrad);
			betaGrad *= 1/this->numSamples;
			l2Regularizer_Gradient(betaTemp, betaGrad); // L2
			l1Regularizer_Subgradient(betaTemp, betaGrad); // L1
			history.gradNormSq.push_back(betaGrad.l2normsq());
			for(int k=0; k<numClass-1; k++)
			{
				interceptGrad[k] /= this->numSamples;
				history.gradNormSq.back()  += interceptGrad[k]*interceptGrad[k];
			}
			// reset timer
			gettimeofday(&start,nullptr) ; // set timer start 
		}

		if (history.gradNormSq.back() <= pow(10,-11))
		{
			std::cout << "first order optimality reached; stop opt process\n";
			break;
		}	
	}

	this->_beta = betaTemp;
	this->_intercept = interceptTemp;

} //end of proximalHybridGD


void LogisticRegression::proximalAGD(double stepSize,
		int maxIter, OptHistory &history, bool writeHistory)
{
	/* 
		solve  min 1/n * sum L(x_i,y_i, beta, intercept) + l1lambda * R(beta) 
	using Accelerated Proximal Gradient Descent 
	*/
	// working variables
	Beta betaTemp(this->_beta);
	Beta betaOld(this->_beta);
	Beta betaAccelerated(this->_beta);
	Beta betaGrad(numClass, dimension, true);
	std::vector<double> interceptTemp(this->_intercept);
	std::vector<double> interceptOld(this->_intercept);
	std::vector<double> interceptAccelerated(this->_intercept);
	std::vector<double> interceptGrad(numClass-1,0.0);
	// opt initialization 
	if (writeHistory)
	{
		history.iterTime.push_back(0.0);
		history.fobj.push_back(this->objValue(betaTemp, interceptTemp));
		for(int n=0; n<numSamples; n++)
			stochasticGradient(n, betaTemp, interceptTemp, betaGrad, interceptGrad);
		betaGrad *= 1/this->numSamples;
		l2Regularizer_Gradient(betaTemp, betaGrad); // L2
		l1Regularizer_Subgradient(betaTemp, betaGrad); // L1
		history.gradNormSq.push_back(betaGrad.l2normsq());
		for(int k=0; k<numClass-1; k++)
		{
			interceptGrad[k] /= this->numSamples;
			history.gradNormSq.back()  += interceptGrad[k]*interceptGrad[k];
		}
	}	

	int countDataPass = 0;
	struct timeval start, finish;	
	gettimeofday(&start,nullptr) ; // set timer start 
	double theta = 1.0, theta_new = 1.0, momentum = 0.0;
	for(int t=0; t<maxIter; t++)
	{
		//std::cout << "***************** iteration " << t << " *****************\n";
		std::fill(interceptGrad.begin(), interceptGrad.end(), 0.0);
		betaGrad.setzero();
		// sample selection and gradient
		for(int n=0; n<numSamples; n++)
			stochasticGradient(n, betaAccelerated, interceptAccelerated, betaGrad, interceptGrad);
		countDataPass += numSamples;
		for(int k=0; k<numClass-1; k++)
		{
			interceptGrad[k] *= (1.0/numSamples);
			interceptTemp[k] -= stepSize * interceptGrad[k];
		}
		betaGrad *= (1.0/numSamples);
		l2Regularizer_Gradient(betaAccelerated, betaGrad); //add gradient wrt L2 term
		// proximal update
		betaTemp -= betaGrad * stepSize;
		proximalL1(betaTemp);
		// acceleration 
		if (this->_l1Lambda > 0)
		{
			theta_new = (1 + sqrt(1 + 4 * theta * theta))/2;
			momentum = (theta - 1)/theta_new;
			theta = theta_new;
		} else 
		{
			momentum = (double)t/(t+3.0);
		}

		betaAccelerated = betaTemp + (betaTemp - betaOld) * momentum;
		betaOld = betaTemp;
		for(int k=0; k<numClass-1; k++)
		{
			interceptAccelerated[k] = interceptTemp[k] + momentum * (interceptTemp[k]-interceptOld[k]);
			interceptOld[k] = interceptTemp[k];
		}
		if (writeHistory & countDataPass%this->numSamples==0)
		{
			// timer
			gettimeofday(&finish,nullptr) ; // set timer start 
			history.iterTime.push_back(history.iterTime.back()+finish.tv_sec-start.tv_sec);     
			history.iterTime.back() += (finish.tv_usec-start.tv_usec)/(1000.0 * 1000.0); 
			// obj value
			history.fobj.push_back(this->objValue(betaTemp, interceptTemp));
			// first order optimality
			std::fill(interceptGrad.begin(), interceptGrad.end(), 0.0);
			betaGrad.setzero();			
			for(int n=0; n<numSamples; n++)
				stochasticGradient(n, betaTemp, interceptTemp, betaGrad, interceptGrad);
			betaGrad *= 1/this->numSamples;
		    l2Regularizer_Gradient(betaTemp, betaGrad); // L2
			l1Regularizer_Subgradient(betaTemp, betaGrad); // L1
			history.gradNormSq.push_back(betaGrad.l2normsq());
			for(int k=0; k<numClass-1; k++)
			{
				interceptGrad[k] /= this->numSamples;
				history.gradNormSq.back()  += interceptGrad[k]*interceptGrad[k];
			}
			// reset timer
			gettimeofday(&start,nullptr) ; // set timer start 
		}

		if (history.gradNormSq.back() <= pow(10,-11))
		{
			std::cout << "first order optimality reached; stop opt process\n";
			break;
		}
	}

	this->_beta = betaTemp;
	this->_intercept = interceptTemp;

} //end of proximalGD
