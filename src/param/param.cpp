#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <Random123/threefry.h>
#include <Random123/philox.h>
#include "uniform.hpp"
#include "matvec.h"
#include "param.h"
#include "common.h"

BlockCholeskey::BlockCholeskey(int numclass, int dim, bool zeroinit): LogisticParam(numclass, dim) 
{
	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = static_cast<int> (time(nullptr));//seed
	for(int i = 0; i<numClass; i++)
	{  
		CSR_matrix factor; 
		factorArray.push_back(factor);
		factorArray[i].number_rows = dimension;
		factorArray[i].number_cols = dimension;
		factorArray[i].row_array_length = dimension+1;
		factorArray[i].nnz = (1+dimension)*dimension / 2;
		if(zeroinit)
		{
			for(int j=0 ; j<factorArray[i].nnz ;j++)
				factorArray[i].val.push_back(0); 
		}
		else
		{
			for(int j=0 ; j<factorArray[i].nnz ;j++)
			{
				ctr[0] = i;
				ctr[1] = j;	
        		CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)
				double x = r123::uneg11<double>(unidrand[0]);
				factorArray[i].val.push_back(x);
			}
		}

		factorArray[i].row_offset.push_back(0);
		factorArray[i].col.push_back(0);
		int row_nnz = 1;
		for(int j=1; j<dimension; j++)
		{
			factorArray[i].row_offset.push_back(factorArray[i].row_offset[j-1] + row_nnz) ; 
			row_nnz = 0;
			for(int k=0; k<=j; k++) 
			{
				factorArray[i].col.push_back(k);
				row_nnz++;
			}
		}
		factorArray[i].row_offset.push_back(factorArray[i].row_offset[dimension-1] + row_nnz);  
	}
}


void BlockCholeskey::setzero()
{
	for(int k=0; k<this->numClass; k++)
		this->factorArray[k].setzero();	
}


double BlockCholeskey::innerProduct(const BlockCholeskey &bchol2) const
{
	try
	{
		if(this->numClass != bchol2.numClass || this->dimension != bchol2.dimension )
			throw "numClass or dimension does not match\n";
	} catch (char const* msg) {
		std::cerr<< msg << std::endl;
		exit(1);
	}
	double prod = 0.0;
	for(int k=0; k<this->numClass; k++)
	{
		for(int i=0; i<this->factorArray[k].nnz; i++)
			prod += this->factorArray[k].val[i]*bchol2.factorArray[k].val[i];
	}	
	return prod;
}

double BlockCholeskey::l2normsq() const
{
	double norm = 0.0;
	for(int k=0; k<this->numClass; k++)
	{
		for(int i=0; i<this->factorArray[k].nnz; i++)
			norm += this->factorArray[k].val[i]
					*this->factorArray[k].val[i];
	}
	return norm;
}



BlockCholeskey& BlockCholeskey::operator-= (BlockCholeskey const &bcholRHS)
{
	try{
		if (this->numClass != bcholRHS.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != bcholRHS.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cerr<< msg << std::endl;
		exit(1);
	}
	for(int k=0; k<this->numClass; k++)
	{
		for(int i=0; i<this->factorArray[k].nnz ; i++)
			this->factorArray[k].val[i] -= bcholRHS.factorArray[k].val[i];
	}
	
	return *this;
}


BlockCholeskey BlockCholeskey::operator- (BlockCholeskey const &bcholRHS) const
{
	try{
		if (this->numClass != bcholRHS.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != bcholRHS.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cout << msg << std::endl;
		exit(1);
	}

	BlockCholeskey res(*this);
	for(int k=0; k<res.numClass; k++)
	{
		for(int i=0; i<res.factorArray[k].nnz ; i++)
			res.factorArray[k].val[i] -= bcholRHS.factorArray[k].val[i];
	}
	
	return res;
}



BlockCholeskey BlockCholeskey::operator+ (BlockCholeskey const &bcholRHS) const
{
	try{
		if (this->numClass != bcholRHS.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != bcholRHS.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cout << msg << std::endl;
		exit(1);
	}

	BlockCholeskey res(*this);
	for(int k=0; k<res.numClass; k++)
	{
		for(int i=0; i<res.factorArray[k].nnz ; i++)
			res.factorArray[k].val[i] += bcholRHS.factorArray[k].val[i];
	}
	
	return res;
}


BlockCholeskey& BlockCholeskey::operator*= (double scalar)
{
	for(int k=0; k<this->numClass; k++)
	{
		for(int i=0; i<this->factorArray[k].nnz ; i++)
			this->factorArray[k].val[i] *= scalar;
	}
	
	return *this;
}



BlockCholeskey BlockCholeskey::operator* (double scalar)
{
	BlockCholeskey res(*this);
	for(int k=0; k<this->numClass; k++)
	{
		for(int i=0; i<this->factorArray[k].nnz ; i++)
			res.factorArray[k].val[i] *= scalar;
	}
	
	return res;
}


bool BlockCholeskey::operator== (BlockCholeskey const &bcholRHS)
{
	if (this->numClass != bcholRHS.numClass)
		return false;
	if (this->dimension != bcholRHS.dimension)
		return false;
	for(int k=0; k<this->numClass; k++)
	{
		for(int i=0; i<this->factorArray[k].nnz ; i++)
		{	
			if (this->factorArray[k].val[i] != bcholRHS.factorArray[k].val[i])
				return false;
		}
	}

	return true;
}



ClassMeans::ClassMeans(int numclass, int dim, bool zeroinit): LogisticParam(numclass, dim)  
{
	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = static_cast<int> (time(nullptr));//seed
	std::vector<double> row;
	for (int i=0; i<numClass; i++)
	{
		meanVectors.push_back(row);
		if (zeroinit)
		{
			for(int j=0; j<dimension; j++)
				meanVectors[i].push_back(0);
		}
		else 
		{
			for(int j=0; j<dimension; j++)
			{
				ctr[0] = i;
				ctr[1] = j;	
        		CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)
				double x = r123::uneg11<double>(unidrand[0]);	
				meanVectors[i].push_back(x);
			}
		}
	} 
}

void ClassMeans::setzero()
{
	for(int k=0; k<this->numClass; k++)
		std::fill(this->meanVectors[k].begin(), 
			this->meanVectors[k].end(), 0.0);
}


double ClassMeans::innerProduct(const ClassMeans &clmeans2) const
{
	try{
		if(this->numClass != clmeans2.numClass || this->dimension != clmeans2.dimension)
			throw "dimension does not matched\n";
	} catch (char const* msg)
	{
		std::cerr << msg << std::endl;
	
	}

	double prod = 0.0;
	for(int k=0; k<this->numClass; k++)
	{
		for(int i=0; i<this->meanVectors[k].size(); i++)
			prod += this->meanVectors[k][i]*clmeans2.meanVectors[k][i];
	}
	return prod;
}


double ClassMeans::l2normsq() const
{
	double norm = 0.0;
	for(int k=0; k<this->numClass; k++)
	{
		for(int i=0; i<this->meanVectors[k].size(); i++)
			norm += this->meanVectors[k][i]*this->meanVectors[k][i];
	}
	return norm;
}



ClassMeans& ClassMeans::operator-= (ClassMeans const &clmsRHS)
{
	try{
		if (this->numClass != clmsRHS.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != clmsRHS.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cout << msg << std::endl;
		exit(1);
	}

	for(int k=0; k<this->numClass; k++)
	{
		for(int j=0; j<this->dimension; j++)
			this->meanVectors[k][j] -= clmsRHS.meanVectors[k][j];
	}

	return *this;
}



ClassMeans ClassMeans::operator- (ClassMeans const &clmsRHS) const
{
	try{
		if (this->numClass != clmsRHS.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != clmsRHS.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cout << msg << std::endl;
		exit(1);
	}

	ClassMeans res(*this);
	for(int k=0; k<res.numClass; k++)
	{
		for(int j=0; j<res.dimension; j++)
			res.meanVectors[k][j] -= clmsRHS.meanVectors[k][j];
	}

	return res;
}


ClassMeans ClassMeans::operator+ (ClassMeans const &clmsRHS) const
{
	try{
		if (this->numClass != clmsRHS.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != clmsRHS.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cout << msg << std::endl;
	}

	ClassMeans res(*this);
	for(int k=0; k<res.numClass; k++)
	{
		for(int j=0; j<res.dimension; j++)
			res.meanVectors[k][j] += clmsRHS.meanVectors[k][j];
	}

	return res;
}


ClassMeans& ClassMeans::operator*= (double scalar)
{
	for(int k=0; k<this->numClass; k++)
	{
		for(int j=0; j<this->dimension; j++)
			this->meanVectors[k][j] *= scalar;
	}

	return *this;
}


ClassMeans ClassMeans::operator* (double scalar)
{
	ClassMeans res(*this);
	for(int k=0; k<this->numClass; k++)
	{
		for(int j=0; j<this->dimension; j++)
			res.meanVectors[k][j] *= scalar;
	}

	return res;
}


Beta::Beta(int numclass, int dim, bool zeroinit): LogisticParam(numclass, dim)  
{
	CommonUtility::CBRNG g;
	CommonUtility::CBRNG::ctr_type ctr = {{}};
    CommonUtility::CBRNG::key_type key = {{}};	
	key[0] = static_cast<int> (time(nullptr));//seed
	std::vector<double> row;
	for (int i=0; i<numClass-1; i++) //loop goes from 0 to numClass-1, standardized with class K
	{
		beta.push_back(row);
		if (zeroinit)
		{
			for(int j=0; j<dimension; j++)
				beta[i].push_back(0);
		}
		else 
		{
			for(int j=0; j<dimension; j++)
			{
				ctr[0] = i;
				ctr[1] = j;	
        		CommonUtility::CBRNG::ctr_type unidrand = g(ctr, key); //unif(-1,1)
				double x = r123::uneg11<double>(unidrand[0]);	
				beta[i].push_back(0.5*(x+1.0));
			}
		}
	} 
}


Beta& Beta::operator-= (Beta const& otherBeta)
{

	try{
		if (this->numClass != otherBeta.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != otherBeta.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cout << msg << std::endl;
		exit(1);
	}

	for(int k=0; k<this->numClass-1; k++)
	{
		for(int j=0; j<this->dimension; j++)
			this->beta[k][j] -= otherBeta.beta[k][j];
	}

	return *this;

}


Beta Beta::operator- (Beta const &otherBeta) const
{
	try{
		if (this->numClass != otherBeta.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != otherBeta.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cout << msg << std::endl;
		exit(1);
	}

	Beta res(*this);
	for(int k=0; k<res.numClass-1; k++)
	{
		for(int j=0; j<res.dimension; j++)
			res.beta[k][j] -= otherBeta.beta[k][j];
	}

	return res;
}


Beta Beta::operator+ (Beta const &otherBeta) const
{
	try{
		if (this->numClass != otherBeta.numClass)
			throw "numClass in LHS and RHS does not match\n";
		if (this->dimension != otherBeta.dimension)
			throw "dimension in LHS and RHS does not match\n";
	} catch (const char* msg) {
		std::cout << msg << std::endl;
		exit(1);
	}

	Beta res(*this);
	for(int k=0; k<res.numClass-1; k++)
	{
		for(int j=0; j<res.dimension; j++)
			res.beta[k][j] += otherBeta.beta[k][j];
	}

	return res;
}


Beta& Beta::operator*= (double scalar)
{
	for(int k=0; k<this->numClass-1; k++)
	{
		for(int j=0; j<this->dimension; j++)
			this->beta[k][j] *= scalar;
	}

	return *this;
}


Beta Beta::operator* (double scalar)
{
	Beta res(*this);
	for(int k=0; k<this->numClass-1; k++)
	{
		for(int j=0; j<this->dimension; j++)
			res.beta[k][j] *= scalar;
	}

	return res;
}


void Beta::setzero()
{
	for (int k=0; k<this->numClass-1; k++)
		std::fill(this->beta[k].begin(), this->beta[k].end(), 0.0);
}


double Beta::l2normsq() const
{
	double norm = 0.0;
	for(int k=0; k<this->numClass-1; k++)
	{
		for(int i=0; i<this->beta[k].size(); i++)
			norm += this->beta[k][i]*this->beta[k][i];
	}
	return norm;
}


double Beta::innerProduct(const Beta &otherBeta) const
{
	try{
		if(this->numClass != otherBeta.numClass || this->dimension != otherBeta.dimension)
			throw "dimension does not matched\n";
	} catch (char const* msg)
	{
		std::cerr << msg << std::endl;
	
	}

	double innerProd = 0.0;
	for(int k=0; k<this->numClass-1; k++)
	{
		for(int i=0; i<this->dimension;i++)
			innerProd += this->beta[k][i] * otherBeta.beta[k][i];
	}
	return innerProd;
}
