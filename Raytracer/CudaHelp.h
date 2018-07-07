#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

#define CUDA_CALL( expr ) \
{ \
	auto status=expr; \
	if ( status != cudaSuccess )  \
	{ \
		std::cout << "CUDA ERROR: " << (status) << std::endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}
