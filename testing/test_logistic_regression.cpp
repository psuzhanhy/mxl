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
#include "logistic_regression.h"
#include "readinput.h"
#include "opthistory.h"

int main(int argc, char **argv) 
{

	std::cout << "tester for LogisticRegression class \n";
	char *input_filename;
    DenseData data;
    int numclass=4;
    input_filename = argv[optind];		
    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);

	LogisticRegression lr = LogisticRegression(xf, data.label, 
            numclass, xf.number_cols, true);

	double stepsize = 0.001 * xf.number_rows;
    int batchSize = 512;
	int maxIter = 1000;
   
	OptHistory sgdHistory(maxIter); 
	bool writeHistory = true;
	lr.fit_by_SGD(stepsize, batchSize, maxIter, sgdHistory, writeHistory);	
    
	char outfilestr[200];
    sprintf (outfilestr, "testlr_result.txt");

	std::ofstream ofs;
	ofs.open(outfilestr,std::ofstream::out | std::ofstream::app);
	for(int t=0; t<sgdHistory.nll.size(); t++)
	{
		ofs << sgdHistory.nll[t] << "," << sgdHistory.gradNormSq[t] << ","
			<< sgdHistory.paramChange[t] << "," << sgdHistory.iterTime[t] << std::endl;

	}
	ofs.close();
    
	return 0;
}
