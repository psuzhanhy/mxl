#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "matvec.h"


double CSR_matrix::rowInnerProduct(const std::vector<double> &denseVec, 
		int vecLength, int rowID) 
{
	/*
	multiplying a row in a CSR matrix with a dense vector
	*/

    if( vecLength != this->number_cols)
        throw "error: vector dimension does not match \n" ;
    

    if (rowID >= this->number_rows )
        throw "error: selected row number is larger than the number of rows in X \n";
        
    double result = 0;
    for(int i = this->row_offset[rowID] ; i<this->row_offset[rowID+1] ; i++)
        result += denseVec[col[i]] * this->val[i] ; 
    
    return result;
}


void CSR_matrix::rowOuterProduct2LowerTri(int rowID, 
		const std::vector<double> &denseRightVec,
		int start, int end, CSR_matrix &lowerTriCSR) 
{
	/*
	TODO: improve generality
	compute the outer product between a row in a CSR matrix and a dense vector
	row (outer) rightvec, only increment results correspond to nonzero entries 
	 in lowerMat (lower triangluer CSR)
	*/

	int subvecLength = end - start + 1;
	try{
		if (start < 0 || end > denseRightVec.size())
			throw "vec start and end index exceed bound\n";

		if(this->number_cols != subvecLength )
			throw "vector dimension does not match, exiting \n" ;
		   
		if (rowID > this->number_rows)
			throw "elected row number exceed number of rows in CSR matrix\n";

	} catch (const char* msg) {
		std::cout << msg << std::endl;	
	}

    for(int i=this->row_offset[rowID] ; i<this->row_offset[rowID+1]; i++)
    {
        for(int j=lowerTriCSR.row_offset[col[i]]; j<lowerTriCSR.row_offset[col[i]+1]; j++)
            lowerTriCSR.val[j] += this->val[i] * denseRightVec[start+lowerTriCSR.col[j]];
    }
}


double CSR_matrix::quadraticForm(const CSR_matrix &leftMat, int rowID, 
		const std::vector<double> &denseRightVec, int start, int end)
{
	/*
	compute quadratic form x' M y 
	x is a row from another CSR_matrix, given leftMat and rowID
	M is the CSR_matrix in the center
    y is a dense vector on right
	*/

	int subvecLength = end - start + 1;
	try
	{
		if (start < 0 || end > denseRightVec.size())
			throw "vec start and end index exceed bound\n";
		if(leftMat.number_cols != this->number_rows)
			throw "error in computing quadratic form: vector dimension does not match\n" ;
		if( subvecLength != this->number_cols)
			throw "error in computing quadratic form: vector dimension does not match\n" ;
   	} catch (const char* msg) {
		std::cout << msg << std::endl;
	}

    double result = 0;
    double row_innerproduct;

    for(int i=leftMat.row_offset[rowID]; i<leftMat.row_offset[rowID+1]; i++)
    {
        row_innerproduct = 0;
        for(int j=this->row_offset[leftMat.col[i]]; j<this->row_offset[leftMat.col[i]+1]; j++ )
           row_innerproduct+= val[j] * denseRightVec[start+col[j]];
        result += row_innerproduct * leftMat.val[i];
    }

    return result;
}


void CSR_matrix::setzero()
{
	std::fill(this->val.begin(), this->val.end(), 0);
}
