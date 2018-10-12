#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "matvec.h"
#include "mxl_gaussblockdiag.h"
#include "readinput.h"
#include "opthistory.h"

int main(int argc, char **argv) 
{

	std::cout << "tester for MxlGaussianBlockDiag class \n";
	char *input_filename;
    DenseData data;
    int numclass=4;
    input_filename = argv[optind];		
    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);
	MxlGaussianBlockDiag gmxl1 = MxlGaussianBlockDiag(xf, data.label,
						 numclass, xf.number_cols, true, 100);

		
	double stepsize_SGD = 0.001 * xf.number_rows;
	double stepsize_APG = 0.0001 * xf.number_rows;
	double scalar = 1.0;
	double momentum = 1.0;
	double shrinkage = 0.5;
	int maxEpochs = 300;
   
	OptHistory sgdHistory(maxEpochs), apgHistory(maxEpochs); 
	//gmxl1.fit_by_SGD(stepsize_SGD,scalar,maxEpochs, sgdHistory);
	gmxl1.fit_by_APG(stepsize_APG, momentum, shrinkage, maxEpochs, apgHistory);	
	
	OptHistory optHistory(apgHistory);
   

	char outfilestr[200];
    sprintf (outfilestr, "APG_t1_r0.001_m1.0_s0.5_R100.o");

	std::ofstream ofs;
	ofs.open(outfilestr,std::ofstream::out | std::ofstream::app);
	for(int t=0; t<optHistory.fobj.size(); t++)
	{
		ofs << optHistory.fobj[t] << "," << optHistory.gradNormSq[t] << ","
			<< optHistory.paramChange[t] << "," << optHistory.iterTime[t] << std::endl;

	}
	ofs.close();

	auto fitlabel = gmxl1.insamplePrediction();
	double correctfit = 0.0;

	std::vector<int> count(numclass,0.0);
	for(int n=0; n<data.label.size(); n++)
		count[data.label[n]]++;
	
	for(int n=0; n<fitlabel.size(); n++)
	{
		try
		{
			if (fitlabel[n] == -1)	
				throw "-1";
		} catch(const char* msg)
		{
			std::cout << msg << std::endl;
		}
		
		//std::cout << n << " " <<  fitlabel[n] << " " << data.label[n] << std::endl;
		if (data.label[n] == 0 && fitlabel[n] == data.label[n])
			correctfit += 1.0;
	}
	double accuracy = correctfit/count[0];
	std::cout << accuracy << std::endl;
	std::cout << static_cast<double>(count[0])/data.label.size() << std::endl;
		
	return 0;
}
