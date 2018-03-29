#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/generator_iterator.hpp>
#include "common.h"

time_t CommonUtility::time_start = time(nullptr);
int CommonUtility::time_start_int = static_cast<int> (CommonUtility::time_start);
boost::mt19937 CommonUtility::rng(CommonUtility::time_start);
boost::random::uniform_real_distribution<> CommonUtility::unid(0, 1);
boost::variate_generator<boost::mt19937&,boost::random::uniform_real_distribution<> > CommonUtility::unid_init(rng, unid);

boost::normal_distribution<> CommonUtility::nd(0.0,1);
boost::variate_generator<boost::mt19937&,boost::normal_distribution<> > CommonUtility::var_nor(rng, nd);

