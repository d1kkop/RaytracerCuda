#pragma once
// Inluce stdio and assert before cuda_runtime otherwise asserts and printf in kernel wont work..
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CALL( expr ) \
{ \
	auto status=expr; \
	if ( status != cudaSuccess )  \
	{ \
		std::cout << "CUDA ERROR: " << (status) << " " << cudaGetErrorString(status) << std::endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}
