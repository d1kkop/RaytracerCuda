#pragma once

#include "Types.h"
// Inluce stdio and assert before cuda_runtime otherwise asserts and printf in kernel wont work..
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include "SharedTypes.h"
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
using namespace glm;
using namespace Beam;


#if CUDA 

    #define GLOBAL __global__
    #define DEVICE __device__
    #define FDEVICE __forceinline__ DEVICE
    #define THREAD_FENCE() __threadfence()
    #define CONSTANT __constant__
    #define INLINE
    #define RESTRICT

    #define bDim blockDim
    #define tIdx threadIdx
    #define bIdx blockIdx

    template <class T> FDEVICE T atomicAdd2(T* t, T v) { return atomicAdd(t, v); }
    template <class T> FDEVICE T atomicCAS2(T* lk, T old, T nw) { return atomicCAS(lk, old, nw); }

#else

    #define GLOBAL
    #define DEVICE
    #define FDEVICE
    #define THREAD_FENCE()
    #define CONSTANT
    #define INLINE inline
    #define RESTRICT __restrict

    static uint4 cpu_blockDim;
    static uint4 cpu_threadIdx;
    static uint4 cpu_blockIdx;
    #define bDim cpu_blockDim
    #define tIdx cpu_threadIdx
    #define bIdx cpu_blockIdx

    template <typename T> T atomicAdd2(T* t, T v) { T old = *t; *t += v; return old; }
    template <typename T> T atomicCAS2(T* lk, T old, T nw) { *lk = nw; return old; }

#endif


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


FDEVICE INLINE i32 _min(i32 a, i32 b) { return a < b ? a : b; }
FDEVICE INLINE i32 _max(i32 a, i32 b) { return a > b ? a : b; }
FDEVICE INLINE u32 _min(u32 a, u32 b) { return a < b ? a : b; }
FDEVICE INLINE u32 _max(u32 a, u32 b) { return a > b ? a : b; }
FDEVICE INLINE float _min(float a, float b) { return a < b ? a : b; }
FDEVICE INLINE float _max(float a, float b) { return a > b ? a : b; }
FDEVICE INLINE vec2 _min(const vec2& a, const vec2& b) { return vec2(_min(a.x, b.x), _min(a.y, b.y)); }
FDEVICE INLINE vec2 _max(const vec2& a, const vec2& b) { return vec2(_max(a.x, b.x), _max(a.y, b.y)); }
FDEVICE INLINE vec3 _min(const vec3& a, const vec3& b) { return vec3(_min(a.x, b.x), _min(a.y, b.y), _min(a.z, b.z)); }
FDEVICE INLINE vec3 _max(const vec3& a, const vec3& b) { return vec3(_max(a.x, b.x), _max(a.y, b.y), _max(a.z, b.z)); }
//template <> vec3 _min(const vec3& a, const vec3& b) { return vec3(_min(a.x, b.x), _min(a.y, b.y), _min(a.z, b.z)); }
//template <> vec3 _max(const vec3& a, const vec3& b) { return vec3(_max(a.x, b.x), _max(a.y, b.y), _max(a.z, b.z)); }


FDEVICE INLINE
u32 rgb(float r, float g, float b)
{
#ifndef CUDA
    u32 ru = lrintf( _max(0.f,_min(255.f,r*255.f)) );
    u32 gu = lrintf( _max(0.f,_min(255.f,g*255.f)) );
    u32 bu = lrintf( _max(0.f,_min(255.f,b*255.f)) );
#else
    u32 ru = u32( _max(0.f,_min(255.f,r*255.f)) );
    u32 gu = u32( _max(0.f,_min(255.f,g*255.f)) );
    u32 bu = u32( _max(0.f,_min(255.f,b*255.f)) );
#endif
    return (ru<<16)|(gu<<8)|bu;
}

FDEVICE INLINE
u32 rgb(vec3 v)
{
    return rgb(v.x,v.y,v.z);
}

FDEVICE INLINE
u32 rgb(float r)
{
#ifndef CUDA
    u32 ru = lrintf( _max(0.f,_min(255.f,r*255.f)) );
#else
    u32 ru = u32( _max(0.f,_min(255.f,r*255.f)) );
#endif
    return (ru<<16)|(ru<<8)|ru;
}

FDEVICE INLINE
float bmTriIntersect( const vec3 &orig, const vec3 &dir, 
                      const vec3 &v0, const vec3 &v1, const vec3 &v2, 
                      float& u, float& v)
{ 
    vec3 v0v1 = v1 - v0; 
    vec3 v0v2 = v2 - v0; 
    vec3 pvec = cross(dir, v0v2); 
    float det = dot(v0v1, pvec); 

//#ifdef CULLING 
//    // if the determinant is negative the triangle is backfacing
//    // if the determinant is close to 0, the ray misses the triangle
//    if (det < kEpsilon) return FLT_MAX;
//#else 
//    // ray and triangle are parallel if det is close to 0
//    if (fabs(det) < kEpsilon) return FLT_MAX; 
//#endif 
//    float invDet = 1.f / det; 
// 

    // IF determinate is small or zero, invDet will be large or inifnity, in either case the the computations remain valid.

    float invDet = 1.f/det;
    vec3 tvec = orig - v0; 
    u = dot(tvec, pvec) * invDet; 
    if ( u<0||u>1 ) return FLT_MAX;

    vec3 qvec = cross(tvec, v0v1); 
    v = dot(dir, qvec) * invDet; 
    if ( v<0||v+u>1 ) return FLT_MAX;

    float dist = (dot(v0v2, qvec)*invDet);
    return dist;

    /*return ( u<0||u>1 ? FLT_MAX : 
            ( v<0||v+u>1 ? FLT_MAX : 
             ( dot(v0v2, qvec)*invDet) ));*/
} 

// https://tavianator.com/fast-branchless-raybounding-box-intersections/
FDEVICE INLINE
float bmBoxRayIntersect(const vec3& bMin, const vec3& bMax,
                        const vec3& orig, const vec3& invDir)
{
    vec3 tMin   = (bMin - orig) * invDir;
    vec3 tMax   = (bMax - orig) * invDir;
    vec3 tMax2  = _max(tMin, tMax);
    float ftmax = _min(tMax2.x, _min(tMax2.y, tMax2.z));
    if ( ftmax < 0.f ) return FLT_MAX;
    vec3 tMin2  = _min(tMin, tMax);
    float ftmin = _max(tMin2.x, _max(tMin2.y, tMin2.z));
    float dist  = _max(0.f, ftmin);
    dist = (ftmax >= ftmin ? dist : FLT_MAX);
    return dist;
}

// https://tavianator.com/fast-branchless-raybounding-box-intersections/
FDEVICE INLINE
float bmBoxRayIntersectNoZero(const vec3& bMin, const vec3& bMax,
                              const vec3& orig, const vec3& invDir)
{
    vec3 tMin   = (bMin - orig) * invDir;
    vec3 tMax   = (bMax - orig) * invDir;
    vec3 tMin2  = _min(tMin, tMax);
    vec3 tMax2  = _max(tMin, tMax);
    float ftmin = _max(tMin2.x, _max(tMin2.y, tMin2.z));
    float ftmax = _min(tMax2.x, _min(tMax2.y, tMax2.z));
    float dist  = (isinf(ftmin)||ftmin<0.f)?ftmax:ftmin;
    return dist;
}

FDEVICE INLINE
bool bmAABBOverlap(const vec3& tMin, const vec3& tMax, const vec3& bMin, const vec3& bMax)
{
    bool b = 
        ( tMin[0] > bMax[0] ? false :
         ( tMin[1] > bMax[1] ? false :
          ( tMin[2] > bMax[2] ? false :
           ( tMax[0] < bMin[0] ? false :
            ( tMax[1] < bMin[1] ? false :
             ( tMax[2] < bMin[2] ? false : true ))))));
    return b;
}

FDEVICE INLINE
bool bmAABBOverlap2(const vec3& tMin, const vec3& tMax, const vec3& bMin, const vec3& bMax)
{
    if ( tMin[0] > bMax[0] ) return false;
    if ( tMin[1] > bMax[1] ) return false;
    if ( tMin[2] > bMax[2] ) return false;
    if ( tMax[0] < bMin[0] ) return false;
    if ( tMax[1] < bMin[1] ) return false;
    if ( tMax[2] < bMin[2] ) return false;
    return true;
}

FDEVICE INLINE
void bmValidateAABB(const vec3& bMin, const vec3& bMax)
{
    u32 dZero = 0;
    dZero += (bMax[0]-bMin[0]<0) ? 1 : 0;
    dZero += (bMax[1]-bMin[1]<0) ? 1 : 0;
    dZero += (bMax[2]-bMin[2]<0) ? 1 : 0;
    if ( dZero==3 )
    {
        printf( "BoxSize: %.f %.f %.f\nbMin %f %f %f | bMax %f %f %f\n", 
                bMax[0]-bMin[0], bMax[1]-bMin[1], bMax[2]-bMin[2],
                bMin[0], bMin[1], bMin[2], bMax[0], bMax[1], bMax[2]);
        assert( false );
    }
}

FDEVICE INLINE
void bmPrintAABB(const vec3& bMin, const vec3& bMax)
{
    printf( "BoxSize: %.f %.f %.f\nbMin %f %f %f | bMax %f %f %f\n", 
            bMax[0]-bMin[0], bMax[1]-bMin[1], bMax[2]-bMin[2],
            bMin[0], bMin[1], bMin[2], bMax[0], bMax[1], bMax[2]);
}






FDEVICE INLINE float bmFaceRayIntersect(bmFace* face, const vec3& eye, const vec3& dir, const StaticMeshData* meshDataPtrs, float& u, float& v)
{
    assert( meshDataPtrs );
    uint4 idx = face->m_index;
    const StaticMeshData* mesh = &meshDataPtrs[idx.w];
    vec3* vp  = (vec3*)mesh->m_vertexData[ VERTEX_DATA_POSITION ];
    assert( mesh->m_vertexDataSizes[ VERTEX_DATA_POSITION ] == 3 );
    return bmTriIntersect( eye, dir, vp[idx.x], vp[idx.y], vp[idx.z], u, v );
}

template <class T>
FDEVICE INLINE T bmFaceInterpolate(bmFace* face, float u, float v, const StaticMeshData* meshDataPtrs, u32 dataIdx)
{
    assert( meshDataPtrs );
    assert( dataIdx < VERTEX_DATA_COUNT );
    uint4 idx = face->m_index;
    const StaticMeshData* mesh = &meshDataPtrs[idx.w];
    const T* vd = (const T*)mesh->m_vertexData[ dataIdx ];
    const T& vd1 = vd[idx.x];
    const T& vd2 = vd[idx.y];
    const T& vd3 = vd[idx.z];
    float w = 1-(u+v);
    return vd1*w + vd2*u + vd3*v;
}

template <typename T>
struct bmStore
{
    T* m_elements;
    u32 m_top;
    u32 m_max;

    FDEVICE T* getNew(u32 cnt=1)
    {
    #if _DEBUG
        if ( m_top+cnt > m_max )
            printf("m_top = %d, _max = %d\n", m_top, m_max);
        assert(m_top+cnt <= m_max);
    #endif
        u32 old = atomicAdd2<u32>(&m_top, cnt);
        memset(m_elements + old, 0, sizeof(T)*cnt);
        return m_elements + old;
    }

    FDEVICE T* get(u32 idx)
    {
        return m_elements + idx;
    }
};
