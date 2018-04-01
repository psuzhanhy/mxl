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

BlockCholeskey::BlockCholeskey(int numclass, int dim, bool zeroinit): MxLParam(numclass, dim) 
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
				factorArray[i].val.push_back(0.5*(x+1.0));
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
		std::cout << msg << std::endl;
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
	}

	BlockCholeskey res(*this);
	for(int k=0; k<res.numClass; k++)
	{
		for(int i=0; i<res.factorArray[k].nnz ; i++)
			res.factorArray[k].val[i] -= bcholRHS.factorArray[k].val[i];
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



ClassMeans::ClassMeans(int numclass, int dim, bool zeroinit): MxLParam(numclass, dim)  
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
				meanVectors[i].push_back(0.5*(x+1.0));
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
	}

	ClassMeans res(*this);
	for(int k=0; k<res.numClass; k++)
	{
		for(int j=0; j<res.dimension; j++)
			res.meanVectors[k][j] -= clmsRHS.meanVectors[k][j];
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
