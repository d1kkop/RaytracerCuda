#include "CudaComon.cuh"
#include "SharedTypes.h"
using namespace Beam;

#define BM_CLEAR_THREADS 256


__global__ void bmKernelClear(uint4* buffer, u32 maxAddr, u32 cv)
{
    u32 addr = blockIdx.x * blockDim.x + threadIdx.x;
    addr = min(addr, maxAddr);
    buffer[addr] = make_uint4(cv, cv,cv, cv);
}


extern "C"
void bmClear(u32* buffer, u32 pitch, u32 width, u32 height, u32 clearValue)
{
    u32 maxAddr  = (pitch/4*height)/4 -1; // pitch is in bytes (/4 because int, again /4 because write 4 ints at a time)
    dim3 blocks ( (maxAddr+BM_CLEAR_THREADS-1)/BM_CLEAR_THREADS );
    dim3 threads( BM_CLEAR_THREADS );
    bmKernelClear<<< blocks, threads >>>
    ( 
       (uint4*)buffer, maxAddr, clearValue
    );
}
