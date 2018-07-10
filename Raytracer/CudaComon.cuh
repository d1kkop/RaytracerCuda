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


__forceinline__ __device__ u32 rgb(float r, float g, float b)
{
    u32 ru = lrintf( max(0.f,min(255.f,r*255.f)) );
    u32 gu = lrintf( max(0.f,min(255.f,g*255.f)) );
    u32 bu = lrintf( max(0.f,min(255.f,b*255.f)) );
    return (ru<<16)|(gu<<8)|bu;
}

__forceinline__ __device__ u32 rgb(vec3 v)
{
    return rgb(v.x,v.y,v.z);
}

__forceinline__ __device__ u32 rgb(float r)
{
    u32 ru = lrintf( max(0.f,min(255.f,r*255.f)) );
    return (ru<<16)|(ru<<8)|ru;
}

__forceinline__ __device__
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
__forceinline__ __device__
float bmBoxRayIntersect(const vec3& bMin, const vec3& bMax,
                        const vec3& orig, const vec3& invDir)
{
    vec3 tMin   = (bMin - orig) * invDir;
    vec3 tMax   = (bMax - orig) * invDir;
    vec3 tMin2  = min(tMin, tMax);
    vec3 tMax2  = max(tMin, tMax);
    float ftmin = max(tMin2.x, max(tMin2.y, tMin2.z));
    float ftmax = min(tMax2.x, min(tMax2.y, tMax2.z));
    float dist  = max(0.f, ftmin);
    dist = (ftmax >= ftmin ? dist : FLT_MAX);
    return dist;

    //float tx1 = (bMin.x - orig.x)*invDir.x;
    //float tx2 = (bMax.x - orig.x)*invDir.z;

    //float tmin = min(tx1, tx2);
    //float tmax = max(tx1, tx2);

    //float ty1 = (bMin.y - orig.y)*invDir.y;
    //float ty2 = (bMax.y - orig.y)*invDir.y;
 
    //tmin = max(tmin, min(ty1, ty2));
    //tmax = min(tmax, max(ty1, ty2));

    //float tz1 = (bMin.z - orig.z)*invDir.z;
    //float tz2 = (bMax.z - orig.z)*invDir.z;

    //tmin = max(tmin, min(tz1, tz2));
    //tmax = min(tmax, max(tz1, tz2));
 
    //float dist = tmin >= 0.f ? tmin : tmax;
    //dist = (dist < 0.f ? FLT_MAX : dist);
    //return dist;
}
