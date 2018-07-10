#include "Util.h"
#include <chrono>
using namespace std;
using namespace chrono;


namespace Beam
{
	double Util::abs_time()
	{
		return sc<double>( duration_cast<duration<double,milli>>( high_resolution_clock::now().time_since_epoch() ).count() );
	}

    double Util::run_time()
    {
        static double begin = abs_time();
        return abs_time() - begin;
    }

    float Util::time()
    {
        return sc<float>( timeD() );
    }

    double Util::timeD()
    {
        return run_time()*0.001;
    }
}