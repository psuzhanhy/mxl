#ifndef common_H
#define common_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <Random123/threefry.h>


class CommonUtility
{
	public:
    	typedef r123::Threefry2x64 CBRNG;
		static time_t time_start;
		static int time_start_int;
		static int numThreads;
};

#endif
