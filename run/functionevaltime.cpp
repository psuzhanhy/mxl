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
    int R = 100;
    int numClass=4;
    int numThreads=1;

    char hostname[100];
    gethostname (hostname, 100);
    std::cout << "hostname " << hostname << std::endl;
	std::cout << "estimating mixed logit model with diagonal Gaussian mixing distribution  \n";
    static struct option long_options[] =
    {
        {"R", required_argument, 0, 'R'},
        {"classes", required_argument, 0, 'c'},
        {"nthreads", required_argument, 0, 't'},
        {0, 0, 0, 0}
    };
    int c=0;
    while (1)
    {
        int option_index = 0;
        c = getopt_long (argc, argv, "R:c:t:o:", long_options, &option_index);
        if (c==-1) break;
        switch (c)
        {
            case 'R':
                R=atoi(optarg);
                break;
            case 'c':
                numClass=(int)atoi(optarg);
                break;
            case 't':
                numThreads=(int)atoi(optarg);
                break;
        }
    }

    if (optind + 1 != argc)
    {
        std::cerr << "Usage: ./runmixedlogit <inputfilename> [option] [value] \n";
        std::cerr << "Options:\n\n";
        std::cerr << "      -t INT        number of threads [1]\n";
        std::cerr << "      -c INT        number of output classes [4]\n";
        std::cerr << "      -R INT        number of draws [1000]\n";
        std::cerr << "      -o STRING     output file prefix (Required)\n";
        return 1;
    }

    input_filename = argv[optind];
    std::cerr << "number of threads : " << numThreads << std::endl;
    std::cerr << "number of output classes :" << numClass << std::endl;
    std::cerr << "number of draws : " << R << std::endl;

    ReadDenseInput(input_filename, &data);   
	CSR_matrix xf = Dense2CSR(data);
	MxlGaussianBlockDiag gmxl1 = MxlGaussianBlockDiag(xf, data.label,
						 numClass, xf.number_cols, false, R);
   
    struct timeval start, finish;
	gettimeofday(&start,nullptr) ; // set timer start 
    for(int t=0; t<10; t++)
        gmxl1.objValue();
    gettimeofday(&finish,nullptr) ; // set timer finish      
    double functionEvalTime = finish.tv_sec - start.tv_sec;
    functionEvalTime += (finish.tv_usec-start.tv_usec)/(1000.0 * 1000.0);
	functionEvalTime /= 10;
    std::cout << "average time for evaluating objective function: " << functionEvalTime << std::endl;    
   
	return 0;
}
