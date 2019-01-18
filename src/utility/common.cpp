#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include "common.h"
#include <boost/random/normal_distribution.hpp>
#include <boost/random.hpp>

//program start time as seed
time_t CommonUtility::time_start = time(nullptr);
int CommonUtility::time_start_int = static_cast<int> (CommonUtility::time_start);
//number of threads
int CommonUtility::numThreadsForIteration = 1;
//number of secondary threads for accounting
int CommonUtility::numSecondaryThreads = 1;
//number of threads for simulation
int CommonUtility::numThreadsForSimulation = 1;
