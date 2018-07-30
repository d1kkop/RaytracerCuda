#include "CudaComon.cuh"
using namespace Beam;


GLOBAL void bmKernelColorGradient(unsigned int* buffer, u32 size)
{

    u32 i = bIdx.x * bDim.x + tIdx.x;
    i = i < size ? i : 0;

    int blockSize = size/6;
    int colorType = i / blockSize;
    float f = (float)(i % blockSize) / (float)blockSize;
    unsigned int c = (unsigned int)(255*f);
    switch ( colorType )
    {
    case 0:
        buffer[i] = c<<16;
        break;

    case 1:
        buffer[i] = c<<8;
        break;

    case 2:
        buffer[i] = c;
        break;

    case 3:
        buffer[i] = (c<<16) | (c<<8);
        break;

    case 4:
        buffer[i] = (c<<8) | c;
        break;

    case 5:
        buffer[i] = (c<<16) | c;
        break;
    }
}



extern "C"
void bmStartColorGradient(u32* buffer, u32 w, u32 h)
{
#if CUDA
    auto size = w*h;
    bmKernelColorGradient<<< (size+255)/256, 256 >>>(buffer, size);
#endif
}