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
	//boost::mt19937 rng(time(nullptr));
	MxlGaussianBlockDiag gmxl1 = MxlGaussianBlockDiag(xf, data.label,
						 numclass, xf.number_cols, false, 100);


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
		
	





	return 0;
}


