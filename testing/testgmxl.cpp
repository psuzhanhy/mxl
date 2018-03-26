#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/generator_iterator.hpp>
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
		
	std::cout << "NLL: " << gmxl1.negativeLogLik() << std::endl;
	//std::cout << "NLL: " << gmxl2.negativeLogLik() << std::endl;







	return 0;
}


