#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "matvec.h"
#include "param.h"
#include "mxl.h"
#include "mxl_gaussblockdiag.h"
#include "rnggenerator.h"
	
MxlGaussianBlockDiag::MxlGaussianBlockDiag(CSR_matrix xf, std::vector<int> lbl,
							int numclass, int dim, bool zeroinit, 
							int numdraws): 
							MixedLogit(xf,lbl,numclass,dim), 
							means(numclass, dim, zeroinit),
							covCholeskey(numclass, dim, zeroinit),
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
			this->classConstants.push_back(0);
	} 
	else
	{				
		//boost::random::uniform_real_distribution<> unid(0, 1);
		//boost::variate_generator<boost::mt19937&,boost::random::uniform_real_distribution<> > unid_init(rng, unid);
		for(int k=0; k<this->numClass; k++)				
			this->classConstants.push_back(RngGenerator::unid_init());
	}
} 



std::vector<double> MxlGaussianBlockDiag::propensityFunction(int sampleID, 
															std::vector<double> normalDraws) 
{
	/*
	* compute propensity score for observation sampleID, and simulated random normal draw
	*/
	    	
	std::vector<double> classPropensity;	  //length: numClass
	for(int k=0; k<this->numClass; k++)
	{
		double meanPropensity = this->xfeature.rowInnerProduct(this->means.meanVectors[k],
								this->means.dimension, sampleID);
		std::vector<double> classdraws;
		for(int i=0; i<this->dimension; i++)
			classdraws.push_back(normalDraws[k*this->dimension+i]);
		double randomPropensity = this->covCholeskey.factorArray[k].quadraticForm(this->xfeature, 
							 	sampleID, classdraws, this->dimension);
        classPropensity.push_back(meanPropensity+randomPropensity+classConstants[k]);
	}
	return classPropensity;
}



std::vector<double> MxlGaussianBlockDiag::multinomialProb(std::vector<double> propensityScore)
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
    for(int i=0; i<this->numClass; i++)
    {
        propensityScore[i] -= maxscore; // avoid overfloat
        propensityScore[i] = exp(propensityScore[i]);
        sumProb += propensityScore[i];
    }
	std::vector<double> classProb;
    for(int i=0; i<numClass; i++)
        classProb.push_back(propensityScore[i]/sumProb);
 
	return classProb;
}



/*
void Stochastic_SAA_MultinomialProb_MLogitGBDCov (ClassMeans Means , BlockCholeskey  CovFactors , double *ClassConstants, CSR_matrix xfeature, int sampleID ,double *NormalDraws ,int R, double *Random_ClassProb)
{
    
    double *ClassMeanPropensity;
    ClassMeanPropensity = (double *) malloc( sizeof(double) * Means.NumClass);
    for(int k=0 ;k<Means.NumClass;k++)
    {
        ClassMeanPropensity[k] = rowInnerProduct( Means.MeanVectors[k] , Means.Dimension , xfeature , sampleID );
        //std::cout << ClassMeanPropensity[k] << " ";
    }
    
    double *ClassCovPropensity; //R-by-NumClass
    ClassCovPropensity = (double *) malloc( sizeof(double) * R * CovFactors.NumClass);
    for(int r=0 ; r<R ; r++ )
    {   
//        ClassCovPropensity[r] = (double *) malloc(sizeof(double) * CovFactors.NumClass);
        for(int k=0 ; k<CovFactors.NumClass ; k++)
        {
            ClassCovPropensity[r*CovFactors.NumClass+k] =quadraticForm(xfeature, sampleID , &NormalDraws[r*CovFactors.NumClass*CovFactors.Dimension+k*CovFactors.Dimension], CovFactors.Dimension , CovFactors.FactorArray[k] );
            //std::cout <<  ClassCovPropensity[r][k] << " ";
        }
        //std::cout << std::endl;
    }

    double *ClassTotalPropensity;
    ClassTotalPropensity = (double *) malloc( sizeof(double) * Means.NumClass);
    for(int r=0 ;r<R ; r++)
    {
        for(int k=0; k<Means.NumClass; k++)
            ClassTotalPropensity[k] = ClassMeanPropensity[k]+ClassCovPropensity[r*CovFactors.NumClass+k]+ClassConstants[k];
        multinomialProb( ClassTotalPropensity, CovFactors.NumClass , &Random_ClassProb[r*Means.NumClass] );
    }

    free(ClassMeanPropensity);
//    for(int r=0; r<R; r++)
//        free(ClassCovPropensity[r]);
    free(ClassCovPropensity);
    free(ClassTotalPropensity);

}

*/

