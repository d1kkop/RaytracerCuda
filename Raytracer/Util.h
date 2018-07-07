#pragma once

#include "Types.h"


namespace Beam
{
	class Util
	{
	public:
        // Elapsed ms since epoch.
		static u64 abs_time();
        // Elapsed ms since first call to run_time.
        static u64 run_time(); 
        // Elapsed seconds sine first call to time.
        static float time();
	};


    template <class To, class From>
    inline To rc(From f) { return reinterpret_cast<To>(f); }

    template <class To, class From>
    inline To sc(From f) { return static_cast<To>(f); }
}