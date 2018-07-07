#include "Util.h"
#include <chrono>
using namespace std;
using namespace chrono;


namespace Beam
{
	u64 Util::abs_time()
	{
		return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
	}

    u64 Util::run_time()
    {
        static u64 begin = abs_time();
        return abs_time() - begin;
    }

    float Util::time()
    {
        return sc<float>(sc<double>(run_time())*0.001);
    }
}