#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/generator_iterator.hpp>
#include "matvec.h"
#include "param.h"
#include "rnggenerator.h"

BlockCholeskey::BlockCholeskey(int numclass, int dim, bool zeroinit): MxLParam(numclass, dim) 
{
	//boost::random::uniform_real_distribution<> unid(0, 1);
	//boost::variate_generator<boost::mt19937&,boost::random::uniform_real_distribution<> > unid_init(rng, unid);
	for(int i = 0; i <numClass ; i++)
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
				factorArray[i].val.push_back(RngGenerator::unid_init());
				//factorArray[i].val.push_back(0.5) ; //for testing		
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


BlockCholeskey& BlockCholeskey::operator+= (BlockCholeskey const& bcholRHS)
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
			this->factorArray[k].val[i] += bcholRHS.factorArray[k].val[i];
	}
	
	return *this;
}


bool BlockCholeskey::operator== (BlockCholeskey const& bcholRHS)
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
	//boost::uniform_real<> unid(0, 1);
	//boost::variate_generator<boost::mt19937&,boost::uniform_real<> > unid_init(rng, unid);
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
				meanVectors[i].push_back(RngGenerator::unid_init());
		}
	} 
}


ClassMeans& ClassMeans::operator+= (ClassMeans const& clmsRHS)
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
			this->meanVectors[k][j] += clmsRHS.meanVectors[k][j];
	}

	return *this;
}
