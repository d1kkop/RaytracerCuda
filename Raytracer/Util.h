#pragma once

#include "Types.h"


namespace Beam
{
	class Util
	{
	public:
		static double abs_time();
        static double run_time(); 
        static float time();
        static double timeD();
	};


    template <class To, class From>
    inline To rc(From f) { return reinterpret_cast<To>(f); }

    template <class To, class From>
    inline To sc(From f) { return static_cast<To>(f); }
}