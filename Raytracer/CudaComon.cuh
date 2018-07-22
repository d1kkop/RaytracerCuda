#pragma once

#include "Types.h"
// Inluce stdio and assert before cuda_runtime otherwise asserts and printf in kernel wont work..
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
using namespace glm;
constexpr float kEpsilon = 0.000001f;


#define CUDA 1

#if CUDA 

    #define GLOBAL __global__
    #define FORCEINLINE __forceinline__
    #define DEVICE __device__
    #define FDEVICE FORCEINLINE DEVICE
    #define THREAD_FENCE() __threadfence()
    #define INLINE

    #define bDim blockDim
    #define tIdx threadIdx
    #define bIdx blockIdx

    template <class T> FDEVICE T atomicAdd2(T* t, T v) { return atomicAdd(t, v); }
    template <class T> FDEVICE T atomicCAS2(T* lk, T old, T nw) { return atomicCAS(lk, old, nw); }

#else

    #define GLOBAL
    #define FORCEINLINE
    #define DEVICE
    #define FDEVICE
    #define THREAD_FENCE()
    #define INLINE inline

    static uint4 cpu_blockDim;
    static uint4 cpu_threadIdx;
    static uint4 cpu_blockIdx;
    #define bDim cpu_blockDim
    #define tIdx cpu_threadIdx
    #define bIdx cpu_blockIdx

    template <typename T> T atomicAdd2(T* t, T v) { T old = *t; *t += v; return old; }
    template <typename T> T atomicCAS2(T* lk, T old, T nw) { *lk = nw; return old; }

#endif


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

#ifdef CULLING 
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < kEpsilon) return FLT_MAX;
#else 
    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < kEpsilon) return FLT_MAX; 
#endif 
    float invDet = 1.f / det; 
 
    vec3 tvec = orig - v0; 
    u = dot(tvec, pvec) * invDet; 
    if (u < 0 || u > 1) return FLT_MAX;
 
    vec3 qvec = cross(tvec, v0v1); 
    v = dot(dir, qvec) * invDet; 
    if (v < 0 || u + v > 1) return FLT_MAX; 
 
    float dist = dot(v0v2, qvec) * invDet;
    return dist;
} 

// https://tavianator.com/fast-branchless-raybounding-box-intersections/
FDEVICE INLINE
float bmBoxRayIntersect(const vec3& bMin, const vec3& bMax,
                        const vec3& orig, const vec3& invDir)
{
    vec3 tMin   = (bMin - orig) * invDir;
    vec3 tMax   = (bMax - orig) * invDir;
    vec3 tMin2  = _min(tMin, tMax);
    vec3 tMax2  = _max(tMin, tMax);
    float ftmin = _max(tMin2.x, _max(tMin2.y, tMin2.z));
    float ftmax = _min(tMax2.x, _min(tMax2.y, tMax2.z));
    float dist  = _max(0.f, ftmin);
    dist = (ftmax >= ftmin ? dist : FLT_MAX);
    return dist;

    //float tx1 = (bMin.x - orig.x)*invDir.x;
    //float tx2 = (bMax.x - orig.x)*invDir.z;

    //float tmin = _min(tx1, tx2);
    //float tmax = _max(tx1, tx2);

    //float ty1 = (bMin.y - orig.y)*invDir.y;
    //float ty2 = (bMax.y - orig.y)*invDir.y;
 
    //tmin = _max(tmin, _min(ty1, ty2));
    //tmax = _min(tmax, _max(ty1, ty2));

    //float tz1 = (bMin.z - orig.z)*invDir.z;
    //float tz2 = (bMax.z - orig.z)*invDir.z;

    //tmin = _max(tmin, _min(tz1, tz2));
    //tmax = _min(tmax, _max(tz1, tz2));
 
    //float dist = tmin >= 0.f ? tmin : tmax;
    //dist = (dist < 0.f ? FLT_MAX : dist);
    //return dist;
}
