#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "matvec.h"


double CSR_matrix::rowInnerProduct(std::vector<double> denseVec, int vecLength, int rowID) 
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


void CSR_matrix::rowOuterProduct2LowerTri(int rowID, std::vector<double> denseRightVec,
										  int rightVecLength, CSR_matrix *lowerTriCSR) 
{
	/*
	TODO: improve generality
	compute the outer product between a row in a CSR matrix and a dense vector
	row (outer) rightvec, only increment results correspond to nonzero entries 
	 in lowerMat (lower triangluer CSR)
	*/

    if(this->number_cols != rightVecLength )
        throw "vector dimension does not match, exiting \n" ;
        

    if (rowID > this->number_rows)
        throw "elected row number exceed number of rows in CSR matrix\n";
        
	/*
	TODO: verify creating of CSR_matrix in LowerMat,
	may cause trouble when entire row is 0
	*/
    for(int i=this->row_offset[rowID] ; i<this->row_offset[rowID+1]; i++)
    {
        for(int j=lowerTriCSR->row_offset[col[i]]; j<lowerTriCSR->row_offset[col[i]+1]; j++)
            lowerTriCSR->val[j] += this->val[i] * denseRightVec[lowerTriCSR->col[j]];
    }
}


double CSR_matrix::quadraticForm(CSR_matrix leftMat, int rowID, std::vector<double> denseRightVec, int rightVecLength)
{
	/*
	compute quadratic form x' M y 
	x is a row from another CSR_matrix, given leftMat and rowID
	M is the CSR_matrix in the center
    y is a dense vector on right
	*/

    if(leftMat.number_cols != this->number_rows)
        throw "error in computing quadratic form: vector dimension does not match\n" ;
                
    if( rightVecLength != this->number_cols)
        throw "error in computing quadratic form: vector dimension does not match\n" ;
   
    double result = 0;
    double row_innerproduct;

    for(int i=leftMat.row_offset[rowID]; i<leftMat.row_offset[rowID+1]; i++)
    {
        row_innerproduct = 0;
        for(int j=this->row_offset[leftMat.col[i]]; j<this->row_offset[leftMat.col[i]+1]; j++ )
           row_innerproduct+= val[j] * denseRightVec[col[j]];
        result += row_innerproduct * leftMat.val[i];
    }

    return result;
}


