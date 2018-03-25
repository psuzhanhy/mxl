#ifndef RNG_GENERATOR_H
#define RNG_GENERATOR_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/generator_iterator.hpp>

class RngGenerator
{
	public:
		static boost::mt19937 rng;
		static boost::random::uniform_real_distribution<> unid;
		static boost::variate_generator<boost::mt19937&,boost::random::uniform_real_distribution<> > unid_init;
};
#endif
