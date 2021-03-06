#ifndef common_H
#define common_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <Random123/threefry.h>
#include <Random123/philox.h>
#include <boost/random/normal_distribution.hpp>
#include <boost/random.hpp>

class CommonUtility
{
	public:
    	//typedef r123::Threefry2x64 CBRNG;
		typedef r123::Philox2x64 CBRNG;
		static time_t time_start;
		static int time_start_int;
		static int numThreadsForIteration;
		static int numSecondaryThreads;
		static int numThreadsForSimulation;
};

#endif
