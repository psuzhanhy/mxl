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
#include <getopt.h>

int main(int argc, char **argv) 
{


    char *input_filename;
    DenseData data;
    std::string alg("SGD");
    int maxEpoch = 10;
    double stepSizeSGD=0.1;
    double stepSizeAGD=0.1;
    std::string sgdStepRule("epochdecreasing");
    int numClass=4;
    int numThreads=1;
    char *oPrefix=NULL;
	double shrinkageAGD = 0.5;
    int batchSize = 1;

    char hostname[100];
    gethostname (hostname, 100);
    std::cout << "hostname " << hostname << std::endl;
	std::cout << "estimating fixed effect logit model \n";
    static struct option long_options[] =
    {
        {"Algorithm", required_argument, 0, 'a'},
        {"epochs", required_argument, 0, 'e'},
        {"initial SGD step-size", optional_argument, 0, 'r'},
        {"AGD step-size", optional_argument, 0, 's'},
        {"SGD step-size rule", optional_argument, 0, 'd'},
        {"classes", required_argument, 0, 'c'},
        {"nthreads", required_argument, 0, 't'},
        {"oprefix", required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };
    int c=0;
    while (1)
    {
        int option_index = 0;
        c = getopt_long (argc, argv, "a:e:r:s:d:c:t:o:", long_options, &option_index);
        if (c==-1) break;
        switch (c)
        {
            case 'a':
                alg=optarg;
                break;
            case 'e':
                maxEpoch=atoi(optarg);
                break;
            case 'r':
                stepSizeSGD=(double)atof(optarg);
                break;
            case 's':
                stepSizeAGD=(double)atof(optarg);
                break;
            case 'd':
                sgdStepRule=optarg;
                break;
            case 'c':
                numClass=(int)atoi(optarg);
                break;
            case 't':
                numThreads=(int)atoi(optarg);
                break;
            case 'o':
                oPrefix = optarg;
                break;
        }
    }

    if ((optind + 1 != argc) || (oPrefix == NULL))
    {
		if (oPrefix == NULL)
	        std::cerr << "\nError:  -o option not provided\n";
        std::cerr << "Usage: ./runmixedlogit <inputfilename> [option] [value] \n";
        std::cerr << "Options:\n\n";
        std::cerr << "      -a STRING     algorithms: SGD, AGD, HYBRID [SGD]\n";
        std::cerr << "      -t INT        number of threads [1]\n";
        std::cerr << "      -c INT        number of output classes [4]\n";
        std::cerr << "      -r REAL       initial SGD step-size [0.1]\n";
        std::cerr << "      -s REAL       AGD step-size [0.1]\n";
        std::cerr << "      -d STRING     SGD step-size Rule [epochdecreasing]\n";
        std::cerr << "      -e INT        number of epochs [10]\n";
        std::cerr << "      -o STRING     output file prefix (Required)\n";
        return 1;
    }

    input_filename = argv[optind];
    std::cerr << "Training settings...\n";
    std::cerr << "Algorithm : " << alg << std::endl; 
    std::cerr << "number of threads : " << numThreads << std::endl;
    std::cerr << "number of output classes :" << numClass << std::endl;
    std::cerr << "SGD step-size (if used): " << stepSizeSGD << std::endl;
    std::cerr << "SGD step-size rule (if used) : " << sgdStepRule << std::endl;
    std::cerr << "AGD step-size (if used): " << stepSizeAGD << std::endl;
    std::cerr << "number of epochs : " << maxEpoch << std::endl;

    CommonUtility::numThreads = numThreads;
    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);
	int p=xf.number_cols;
    int n=xf.number_rows;
	double l1Lambda = 0.0;
	double l2Lambda = 0.0;

	LogisticRegression lr = LogisticRegression(xf, data.label, 
            numClass, p, l1Lambda, l2Lambda, true);

	OptHistory optHistory(maxEpoch); 
    
    if (alg == "AGD")
    {
	    lr.proximalAGD(stepSizeAGD, maxEpoch, optHistory, true);	
    } else if (alg == "SGD")
    {
        int maxIter = maxEpoch * n / batchSize;
	    lr.proximalSGD(stepSizeSGD, sgdStepRule, batchSize, maxIter, optHistory, true, false);	
    } else if (alg == "HYB" || alg == "HYBRID" || alg == "hybrid")
    {
        lr.hybridFirstOrder(stepSizeSGD, batchSize, sgdStepRule, stepSizeAGD,
		    maxEpoch, optHistory);
    } else 
    {
        std::cerr << "unrecognized -a input. Available algorithms: SGD, AGD, HYBRID \n";
        exit(1);
    }
	
	char outfilestr[200];
    char paramfile[200];
   
    sprintf (outfilestr, "%s_%s_t%d_r%.3lf_%s_s%.3lf_e%d.txt", oPrefix, alg.c_str(), numThreads, stepSizeSGD, sgdStepRule.c_str(), stepSizeAGD, maxEpoch);
    sprintf (paramfile, "%s_%s_t%d_r%.3lf_%s_s%.3lf_e%d_param.txt", oPrefix, alg.c_str(), numThreads, stepSizeSGD, sgdStepRule.c_str(), stepSizeAGD, maxEpoch);

    std::cerr << "output file name: " << outfilestr << std::endl;
	std::ofstream ofs(outfilestr);
    if (ofs.fail())
    {
        std::cerr << "Failed to open outputfile.\n";
        exit(1);
    }

	for(int t=0; t<optHistory.fobj.size(); t++)
	{
		ofs << optHistory.fobj[t] << "," << optHistory.gradNormSq[t] << ","
			<< optHistory.iterTime[t] << std::endl;

	}
	ofs.close();
    ofs.clear();
    ofs.open(paramfile);
    if (ofs.fail())
    {
        std::cerr << "Failed to open param outputfile.\n";
        exit(1);
    }

    std::vector<double> constants = lr.getIntercept();
    Beta beta = lr.getBeta();

    for(int k=0; k<constants.size(); k++)
        ofs << constants[k] << std::endl;
    for(int k=0; k<beta.beta.size(); k++)
    {
        for(int i=0; i<beta.beta[k].size(); i++)
            ofs << beta.beta[k][i] << std::endl;
    }  
	ofs.close();	

	return 0;
}
