#include "CudaComon.cuh"
using namespace Beam;


DEVICE float square( vec2 uv, vec2 hsize )
{
    vec2 dv = abs(uv) - hsize;
    float t = _min(0.f, _max(dv.x, dv.y));
    float l = length(_max(dv,vec2(0.f)));
    return float(t + l);
}

DEVICE float rsquare( vec2 uv, vec2 hsize, float edge )
{
    float d =square(uv,hsize);
    return d-edge;
}

FDEVICE vec2 rotate2(vec2 v, float ang)
{
    float s = sin(ang);
    float c = cos(ang);
    return vec2( c*v.x-s*v.y, s*v.x+c*v.y );
}


GLOBAL void bmKernelBlob(u32* buffer, u32 w, u32 h, float time)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = w*h;
    i = _min(i,size);

    // vec2 uv = vec2( (i%w) - (w/2), (i/w) - (h/2) );
    vec2 uv = vec2( (i%w), (i/w) );
    uv -= vec2( w/2, h/2 );

    
    uv = rotate( uv, time );
    uv.y *= 2.f;

 //   uv *= (sin ( time ) + 1.f) *.2 + .3f;

    //float d = length(uv);
    float d = rsquare( uv, vec2(100.f, 100.f), 0.f );
   
    float f = smoothstep( -1.f, 1.f, d );
    f = 1.f-f;

    vec3 bg( 1.f, 1.f, 1.f );
    float s = 1.f- clamp(d / (1500), 0.f, 1.f);
    bg *= pow( s , 2.f );
    vec3 c( 1.f, 0.f, 0.f );

    vec3 m = mix( bg, c, f );


    buffer[i] = rgb( m );
 
}


extern "C"
void bmStartBlob(u32* buffer, int w, int h, float time)
{
#if CUDA
    auto size = w*h;
    bmKernelBlob<<< (size+255)/256, 256 >>>(buffer, w, h, time);
#endif
}