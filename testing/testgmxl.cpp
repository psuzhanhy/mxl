#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include "matvec.h"
#include "mxl_gaussblockdiag.h"
#include "readinput.h"
#include "opthistory.h"

int main(int argc, char **argv) 
{


	char *input_filename;
    DenseData data;
    int numclass=4;
    input_filename = argv[optind];		
    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);
	MxlGaussianBlockDiag gmxl1 = MxlGaussianBlockDiag(xf, data.label,
						 numclass, xf.number_cols, false, 100);

		
	double stepsize_SGD = 1.0 * xf.number_rows;
	double stepsize_APG = 0.01 * xf.number_rows;
	double scalar = 1.0;
	double momentum = 1.0;
	double shrinkage = 0.9;
	int maxEpochs = 10;
   
	OptHistory sgdHistory(maxEpochs), apgHistory(maxEpochs); 
	gmxl1.fit_by_SGD(stepsize_SGD,scalar,maxEpochs, sgdHistory);
	gmxl1.fit_by_APG(stepsize_APG, momentum, shrinkage, maxEpochs, apgHistory);	
    
	return 0;
}


