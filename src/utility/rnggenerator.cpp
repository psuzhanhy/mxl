#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/generator_iterator.hpp>
#include "rnggenerator.h"

boost::mt19937 RngGenerator::rng(time(nullptr));
boost::random::uniform_real_distribution<> RngGenerator::unid(0, 1);
boost::variate_generator<boost::mt19937&,boost::random::uniform_real_distribution<> > RngGenerator::unid_init(rng, unid);

