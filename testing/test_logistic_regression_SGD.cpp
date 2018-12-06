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
#include "param.h"
#include "readinput.h"
#include "opthistory.h"

int main(int argc, char **argv) 
{

	std::cout << "tester for LogisticRegression fit with Stochastic Gradient \n";
	char *input_filename;
    DenseData data;
    int numclass=4;
    input_filename = argv[optind];		
    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);
	int p=xf.number_cols;
	double l1Lambda = 0.0;
	double l2Lambda = 0.1;

	LogisticRegression lr = LogisticRegression(xf, data.label, 
            numclass, p, l1Lambda, l2Lambda, true);

	double stepsize = 0.1;
    int batchSize = 1;
	int maxIter = 10000000;
   
	OptHistory sgdHistory(maxIter); 
	bool writeHistory = true;
	lr.proximalSGD(stepsize, "decreasing", batchSize, maxIter, sgdHistory, writeHistory, false);	
    
	char outfilestr[200];
    sprintf (outfilestr, "testlr_SGD.txt");

	std::ofstream ofs;
	ofs.open(outfilestr,std::ofstream::out | std::ofstream::app);
	
	for(int t=0; t<sgdHistory.fobj.size(); t++)
	{
		ofs << sgdHistory.fobj[t] << "," << sgdHistory.gradNormSq[t] << ","
			<< sgdHistory.iterTime[t] << std::endl;

	}
	
	ofs.close();
    
	Beta betafit = lr.getBeta();
	for(int k=0; k<numclass-1; k++)
	{
		for(int i=0; i<p ;i++)
			std::cout << betafit.beta[k][i] << " ";
		std::cout << std::endl;
	}
	return 0;
}
