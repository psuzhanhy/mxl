#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include "matvec.h"
#include "mxl_gaussblockdiag.h"
#include "readinput.h"

int main(int argc, char **argv) 
{


	char *input_filename;
    DenseData data;
    int numclass=4;
    input_filename = argv[optind];		
    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);
	MxlGaussianBlockDiag gmxl1 = MxlGaussianBlockDiag(xf, data.label,
						 numclass, xf.number_cols, false, 1000);

	/*
	MxlGaussianBlockDiag gmxl2 = MxlGaussianBlockDiag(xf, data.label,
						 numclass, xf.number_cols, false, 100);

	auto choleskey1 = gmxl1.getCovCholeskey();
	auto  choleskey2 = gmxl2.getCovCholeskey();
	
	if (choleskey1 == choleskey2)
	{
		std::cout << "same \n";
	} else {
		std::cout << "not same \n";	
	}
	*/	

	//auto prob = gmxl1.simulatedProbability(0);
	
	//for(auto val : prob)
	//	std::cout << val << std::endl;
	
	struct timeval start, finish;
    gettimeofday(&start,nullptr) ; // set timer start      
	std::cout << "Initial NLL: " << gmxl1.negativeLogLik() << std::endl;
    gettimeofday(&finish,nullptr) ; // set timer finish      
	double elapsedTime = finish.tv_sec - start.tv_sec;      // sec 
    elapsedTime += (finish.tv_usec - start.tv_usec)/(1000.0 * 1000.0);   // us to sec
	std::cout << "time for computing NLL: " << elapsedTime << std::endl;
		
	double stepsize = 1.0;
	double scalar = 1.0;
	int maxEpochs = 1;
    gettimeofday(&start,nullptr) ; // set timer start      
	gmxl1.fit(stepsize,scalar,maxEpochs);
    gettimeofday(&finish,nullptr) ; // set timer finish      
	elapsedTime = finish.tv_sec - start.tv_sec;      // sec 
    elapsedTime += (finish.tv_usec - start.tv_usec)/(1000.0 * 1000.0);   // us to sec
	std::cout << "time for one SGD epoch : " << elapsedTime << std::endl;
	std::cout << "NLL after " << maxEpochs << " epochs: "  << gmxl1.negativeLogLik() << std::endl;


	return 0;
}


