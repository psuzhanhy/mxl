#ifndef MATVEC_H
#define MATVEC_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/generator_iterator.hpp>

struct CSR_matrix  
{
	std::vector<int> row_offset;
    std::vector<double> val;
    std::vector<int> col;
    
    int row_array_length;  // equals to number of rows plus 1
    int number_rows;    // number of rows
    int number_cols;
    int nnz;
 
	~CSR_matrix(){}

	void setzero();

	double rowInnerProduct(const std::vector<double> &denseVec, 
			int vecLength, int rowID);
	
	void rowOuterProduct2LowerTri(int rowID, const std::vector<double> &denseRightVec,
			int start, int end, CSR_matrix &lowerTriCSR);	
	
	double quadraticForm(const CSR_matrix &leftMat, int rowID, 
			const std::vector<double> &denseRightVec, int start, int end);
};



struct  SparseVector 
{ 
	std::vector<double> val;
	std::vector<int> idx;
    int length;    // vector length
    int nnz;
	~SparseVector(){}
} ;


struct DenseData 
{
  std::vector<std::vector<double> > featureMat;
  std::vector<int> label;
};


#endif



