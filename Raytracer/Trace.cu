#include "CudaComon.cuh"
#include "SharedTypes.h"
#include <iostream>
using namespace Beam;
using namespace std;

//constexpr float kEpsilon = 0.000001f;

//// Vectormath https://github.com/erwincoumans/sce_vectormath
//class Ray
//{
//public:
//    Ray();
//public:
//    Vectormath::Aos::Vector3 m_pos;
//    Vectormath::Aos::Vector3 m_dir;
//    Vectormath::Aos::Vector3 m_invDir; // 1.0f / m_dir per elem
//    float m_min;
//    float m_max;
//};
//
//bool BBox::Intersect(const Ray &ray) const
//{
//    using namespace Vectormath;
//    Aos::Vector3 t1(Aos::mulPerElem(m_min - ray.m_pos, ray.m_invDir));
//    Aos::Vector3 t2(Aos::mulPerElem(m_max - ray.m_pos, ray.m_invDir));
//
//    Aos::Vector3 tmin1(Aos::minPerElem(t1, t2));
//    Aos::Vector3 tmax1(Aos::maxPerElem(t1, t2));
//
//    float tmin = Aos::maxElem(tmin1);
//    float tmax = Aos::minElem(tmax1);
//
//    return tmax >= std::_max(ray.m_min, tmin) && tmin < ray.m_max;
//}



FDEVICE
bool intersect( 
    const vec3 &orig, const vec3 &dir, 
    const vec3 &v0, const vec3 &v1, const vec3 &v2, 
    float &dist, float& u, float& v, float& w)
{ 
    vec3 v0v1 = v1 - v0; 
    vec3 v0v2 = v2 - v0; 
    vec3 pvec = cross(dir, v0v2); 
    float det = dot(v0v1, pvec); 

#ifdef CULLING 
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < kEpsilon) return false; 
#else 
    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < kEpsilon) return false; 
#endif 
    float invDet = 1 / det; 
 
    vec3 tvec = orig - v0; 
    u = dot(tvec, pvec) * invDet; 
    if (u < 0 || u > 1) return false; 
 
    vec3 qvec = cross(tvec, v0v1); 
    v = dot(dir, qvec) * invDet; 
    if (v < 0 || u + v > 1) return false; 
 
    dist = dot(v0v2, qvec) * invDet;
    w = 1.f-u-v;
 
    return true; 
} 


GLOBAL void bmKernelTrace(u32* buffer,
                              vec3 eye, mat3 orient, const vec3* initialRays, u32 numRays,
                              const StaticMeshData* meshDataPtrs, u32 numMeshes, u32 time )
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    i = _min(i, numRays-1);

    vec3 dir = initialRays[i];
    dir = orient*dir;

    u32 meshIdx    = (u32) -1;
    uint3 faceIdx  = make_uint3(0,0,0);
    float3 uvw = make_float3(0,0,0);
    
    u32 finalColor = 50;

    // intersect test
//#pragma unroll
    for ( u32 i=0; i<numMeshes; i++ )
    {
        const StaticMeshData* mPtr = meshDataPtrs + i;
        u32*    indices  = mPtr->m_indices;
        float*  vertices = mPtr->m_vertexData[VERTEX_DATA_POSITION]; 
        u32 vertexSize   = mPtr->m_vertexDataSizes[VERTEX_DATA_POSITION]; 
        //11112
   // #pragma unroll
        for ( u32 f=0; f<10; f+=3)
        {
            uint3 fidx = *(uint3*)(indices + f);
            vec3 v1 = *(vec3*)(vertices + fidx.x*vertexSize);
            vec3 v2 = *(vec3*)(vertices + fidx.y*vertexSize);
            vec3 v3 = *(vec3*)(vertices + fidx.z*vertexSize);

            //float dist, u, v, w;
            //if ( intersect(eye, dir, v1, v2, v3, dist, u, v, w) )
            //{
            //    if ( dist < fClosest )
            //    {
            //        fClosest = dist;
            //        meshIdx = i;
            //        faceIdx = fidx;
            //        uvw = make_float3(u,v,w);
            //    }
            //}

            
            //finalColor += 50;
       }
    }


    if ( meshIdx != (u32)-1 )
    {
        auto* mPtr = meshDataPtrs + meshIdx;

        float* normal  = mPtr->m_vertexData[VERTEX_DATA_NORMAL];
        u32 normalSize = mPtr->m_vertexDataSizes[VERTEX_DATA_NORMAL];
     //   float* color   = mPtr->m_vertexData[VERTEX_DATA_EXTRA4];
     //   u32 colorSize  = mPtr->m_vertexDataSizes[VERTEX_DATA_EXTRA4];

    //    vec3 c1 = *(vec3*)(color + faceIdx.x*colorSize);
    //    vec3 c2 = *(vec3*)(color + faceIdx.y*colorSize);
    //    vec3 c3 = *(vec3*)(color + faceIdx.z*colorSize);

        vec3 n1 = *(vec3*)(normal + faceIdx.x*normalSize);
        vec3 n2 = *(vec3*)(normal + faceIdx.y*normalSize);
        vec3 n3 = *(vec3*)(normal + faceIdx.z*normalSize);

        vec3 n = n1*uvw.x + n2*uvw.y + n3*uvw.z;
      //  n = normalize(n);

    //    vec3 sumColor = c1*uvw.x + c2*uvw.y + c3*uvw.z;
    //    finalColor = rgb( sumColor );
        finalColor = rgb( abs(n.z) );

    //    finalColor = (u32)-1;
    }

    buffer[i] = finalColor;
  
}


GLOBAL void bmKernelClear2(u32* buffer, u32 maxAddr, u32 clearColor)
{
    u32 addr = blockIdx.x * blockDim.x + threadIdx.x;
    addr = _min(addr, maxAddr-1);
    uint4* bptr = (uint4*)buffer;
    bptr[addr] = make_uint4(clearColor, clearColor,clearColor, clearColor);
    //buffer[addr] = clearColor;
}


extern "C"
void bmStartTrace(u32* buffer, u32 pitch, u32 width, u32 height,
                  const vec3& eye, const mat3& orient, const vec3* initialRays,
                  const StaticMeshData* meshData, u32 numMeshes)
{

#if CUDA
    u32 maxAddr  = width*height;///4-1;
    auto numRays = maxAddr;
    static u32 i = 0;
    i++;
    bmKernelTrace<<< (numRays+1023)/1024, 1024 >>>( buffer, eye, orient, initialRays, numRays, meshData, numMeshes, i );
#endif

  //  bmKernelClear2<<< (numRays+1023)/1024, 1024 >>>( buffer, pitch, width, 100 );
  //  bmKernelClear<<< (numRays+1023)/1024, 1024  >>>( buffer, maxAddr, 100<<16 );

}
