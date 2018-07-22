#include "CudaComon.cuh"
#include "SharedTypes.h"
using namespace Beam;

#define BM_CLEAR_THREADS 256


GLOBAL void bmKernelClear(u32* buffer, u32 maxAddr, u32 cv)
{
    //if ( threadIdx.x == 0 )
    //    printf("Hello from kernel\n");
    u32 addr = bIdx.x * bDim.x + tIdx.x;
    addr = _min(addr, maxAddr);
    buffer[addr] = cv;
   // buffer[addr] = cv;// make_uint4(cv, cv,cv, cv);
}


extern "C"
void bmClear(u32* buffer, u32 pitch, u32 width, u32 height, u32 clearValue)
{
    u32 maxAddr  = (pitch/4*height) -1; // pitch is in bytes (/4 because int, again /4 because write 4 ints at a time)
#if CUDA
    dim3 blocks ( (maxAddr+BM_CLEAR_THREADS-1)/BM_CLEAR_THREADS );
    dim3 threads( BM_CLEAR_THREADS );
 //   uint4 cv = make_uint4(clearValue, clearValue, clearValue, clearValue);
    bmKernelClear<<< blocks, threads >>>
    ( 
      buffer, maxAddr, clearValue
    );
#else
    bDim.x      = BM_CLEAR_THREADS;
    u32 blocks  = ( (maxAddr+BM_CLEAR_THREADS-1)/BM_CLEAR_THREADS );
    u32 threads = BM_CLEAR_THREADS;
    for ( u32 b=0; b< blocks; b++ )
    {
        bIdx.x = b;
        for ( u32 t=0; t<threads; t++ )
        {
            tIdx.x = t;
            bmKernelClear
            ( 
                buffer, maxAddr, clearValue
            );
        }
    }
#endif
}
