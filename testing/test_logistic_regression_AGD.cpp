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

	std::cout << "tester for LogisticRegression fit with Full Gradient Descent \n";
	char *input_filename;
    DenseData data;
    int numclass=4;
    input_filename = argv[optind];		
    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);
	int p=xf.number_cols;
	double l1Lambda = 0.0;

	LogisticRegression lr = LogisticRegression(xf, data.label, 
            numclass, p, l1Lambda, true);

	double stepsize = 0.1;
	int maxIter = 10000;
   
	OptHistory optHistory(maxIter); 
	bool writeHistory = true;
	lr.proximalGD(stepsize, maxIter, optHistory, writeHistory);	
    
	char outfilestr[200];
    sprintf (outfilestr, "testlr_fitAGD.txt");

	std::ofstream ofs;
	ofs.open(outfilestr,std::ofstream::out | std::ofstream::app);
	for(int t=0; t<optHistory.fobj.size(); t++)
	{
		ofs << optHistory.fobj[t] << "," << optHistory.gradNormSq[t] << ","
		    << optHistory.iterTime[t] << std::endl;

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
