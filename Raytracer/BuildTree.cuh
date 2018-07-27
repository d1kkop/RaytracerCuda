#pragma once
#ifndef BUILD_TREE_CUH
#define BUILD_TREE_CUH
#include "CudaComon.cuh"
#include "SharedTypes.h"
#include <cassert>
using namespace std;
using namespace Beam;

// thread sizes
#define BUILD_TREE_THREADS 256
#define MARCH_THREADS 256

// tree memory
#define BUILD_TREE_MAX_DEPTH 38
#define TREE_SEARCH_DEPTH BUILD_TREE_MAX_DEPTH
#define MAX_FACES_PER_BOX 256
#define MIN_LEAF_SIZE .03f

// Hash world
#define MAX_HASH_ELEMENTS 65536UL




struct bmMaterial
{
    int texture;
};


struct bmFace
{
    uint4 m_index; // x,y,z indices, w meshIdx
    bmMaterial* m_material;
};


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
};


struct bmTreeNode
{
    bmTreeNode* m_left;
    bmTreeNode* m_right;
    bmFace** m_faces;
    u32 m_faceInsertIdx;

    FDEVICE void split(bmStore<bmTreeNode>* store);
    FDEVICE void split2(bool b1, bool b2, bmStore<bmTreeNode>* store);
    FDEVICE void insertFace(bmStore<bmFace>* faceStore, bmStore<bmFace*>* faceGroupStore, u32 meshIdx, uint3 faceIdx, bmMaterial* mat);
};


struct bmStackNode
{
    vec3 m_min, m_max;
    bmTreeNode* m_node;
    u32 m_depth;
    u32 m_splitAxis;

    FDEVICE void init(bmTreeNode* node, const vec3& bMin, const vec3& bMax, u32 splitAxis, u32 depth);
    FDEVICE bool intersect(const vec3& triMin, const vec3& triMax);
};

FDEVICE INLINE float bmFaceRayIntersect(bmFace* face, const vec3& eye, const vec3& dir, const StaticMeshData* meshDataPtrs, float& u, float& v)
{
    assert( meshDataPtrs );
    uint4 idx = face->m_index;
    const StaticMeshData* mesh = &meshDataPtrs[idx.w];
    vec3* vp  = (vec3*)mesh->m_vertexData[ VERTEX_DATA_POSITION ];
    assert( mesh->m_vertexDataSizes[ VERTEX_DATA_POSITION ] == 3 );
    return bmTriIntersect( eye, dir, vp[idx.x], vp[idx.y], vp[idx.z], u, v );
}

FDEVICE INLINE vec4 bmFaceInterpolate(bmFace* face, float u, float v, const StaticMeshData* meshDataPtrs, u32 dataIdx)
{
    assert( meshDataPtrs );
    assert( dataIdx < VERTEX_DATA_COUNT );
    uint4 idx = face->m_index;
    const StaticMeshData* mesh = &meshDataPtrs[idx.w];
    float* vd = mesh->m_vertexData[ dataIdx ];
    u32 dsize = mesh->m_vertexDataSizes[ dataIdx ];
    assert( dsize > 0 && dsize < 4 );
    float* vd1 = vd + idx.x*dsize;
    float* vd2 = vd + idx.y*dsize;
    float* vd3 = vd + idx.z*dsize;
    float w = 1-(u+v);
    switch ( dsize )
    {
    case 1:
        return vec4(vd1[0]*u + vd2[0]*v + vd3[0]*w, 0.f, 0.f, 0.f);
    case 2:
        return vec4(vd1[0]*u + vd2[0]*v + vd3[0]*w, 
                    vd1[1]*u + vd2[1]*v + vd3[1]*w, 0.f, 0.f);
    case 3:
        return vec4(vd1[0]*u + vd2[0]*w + vd3[0]*w, 
                    vd1[1]*u + vd2[1]*w + vd3[1]*w,
                    vd1[2]*u + vd2[2]*w + vd3[2]*w, 0.f);
    case 4:
        return vec4(vd1[0]*u + vd2[0]*v + vd3[0]*w, 
                    vd1[1]*u + vd2[1]*v + vd3[1]*w,
                    vd1[2]*u + vd2[2]*v + vd3[2]*w,
                    vd1[3]*u + vd2[3]*v + vd3[3]*w);
    }
    return vec4(0.f);
}

#endif