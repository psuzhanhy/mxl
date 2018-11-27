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
#include <getopt.h>

int main(int argc, char **argv) 
{


    char *input_filename;
    DenseData data;
    std::string alg("SGD");
    int R = 1000;
    int maxEpoch = 10;
    double stepsize=1.0;
    double momentum=1.0;
    int numClass=4;
    int numThreads=1;
    char *oPrefix=NULL;
	double shrinkage = 0.0001;

    char hostname[100];
    gethostname (hostname, 100);
    std::cout << "hostname " << hostname << std::endl;
	std::cout << "estimating mixed logit model with diagonal Gaussian mixing distribution  \n";
    static struct option long_options[] =
    {
        {"Algorithm", required_argument, 0, 'a'},
        {"R", required_argument, 0, 'R'},
        {"epochs", required_argument, 0, 'e'},
        {"step-size", required_argument, 0, 'r'},
        {"momentum", optional_argument, 0, 'm'},
        {"classes", required_argument, 0, 'c'},
        {"nthreads", required_argument, 0, 't'},
        {"oprefix", required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };
    int c=0;
    while (1)
    {
        int option_index = 0;
        c = getopt_long (argc, argv, "a:R:e:r:m:c:t:o:", long_options, &option_index);
        if (c==-1) break;
        switch (c)
        {
            case 'a':
                alg=optarg;
                break;
            case 'R':
                R=atoi(optarg);
                break;
            case 'e':
                maxEpoch=atoi(optarg);
                break;
            case 'r':
                stepsize=(double)atof(optarg);
                break;
            case 'm':
                momentum=(double)atof(optarg);
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
        std::cerr << "      -r REAL       step-size [1.0]\n";
        std::cerr << "      -r REAL       Nesterov momentum [1.0]\n";
        std::cerr << "      -e INT        number of epochs [10]\n";
        std::cerr << "      -R INT        number of draws [1000]\n";
        std::cerr << "      -o STRING     output file prefix (Required)\n";
        return 1;
    }

    input_filename = argv[optind];
    std::cerr << "Training settings...\n";
    std::cerr << "Algorithm : " << alg << std::endl; 
    std::cerr << "number of threads : " << numThreads << std::endl;
    std::cerr << "number of output classes :" << numClass << std::endl;
    std::cerr << "step-size : " << stepsize << std::endl;
    std::cerr << "number of epochs : " << maxEpoch << std::endl;
    std::cerr << "number of draws : " << R << std::endl;

    CommonUtility::numThreads = numThreads;
    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);
	MxlGaussianBlockDiag gmxl1 = MxlGaussianBlockDiag(xf, data.label,
						 numClass, xf.number_cols, false, 100);
   
	OptHistory optHistory(maxEpoch); 
    
    if (alg == "AGD")
    {
	    gmxl1.fit_by_APG(stepsize, momentum, shrinkage, maxEpoch, optHistory, true);	
    } else if (alg == "SGD")
    {
	    gmxl1.fit_by_SGD(stepsize, maxEpoch, optHistory,  true);        
    } else 
    {
        std::cerr << "unrecognized -a input. Available algorithms: SGD, ALG, HYBRID \n";
    }
	
	char outfilestr[200];
    sprintf (outfilestr, "%s_%s_t%d_r%.1lf_e%d.txt", oPrefix, alg.c_str(), numThreads, stepsize, maxEpoch);
    std::cout << "output file name: " << outfilestr << std::endl;
	std::ofstream ofs(outfilestr);
    if (ofs.fail())
    {
        std::cout << "Failed to open outputfile.\n";
    }

	for(int t=0; t<optHistory.fobj.size(); t++)
	{
		ofs << optHistory.fobj[t] << "," << optHistory.gradNormSq[t] << ","
			<< optHistory.iterTime[t] << std::endl;

	}
	ofs.close();
	
	return 0;
}
