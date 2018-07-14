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
#define BUILD_TREE_MAX_DEPTH 12
#define MAX_FACES_PER_BOX 2


struct bmMaterial
{
    int texture;
};


struct bmFace
{
    uint4 m_index; // x,y,z indices, w meshIdx
    bmMaterial* m_material;
    __device__ float intersect(const vec3& eye, const vec3& dir, const StaticMeshData* meshDataPtrs, u32 numMeshes, float& u, float& v);
};


template <typename T>
struct bmStore
{
    T* m_elements;
    u32 m_top;
    u32 m_max;

    __forceinline__ __device__ T* getNew(u32 cnt=1)
    {
    #if _DEBUG
        if ( m_top+cnt > m_max )
            printf("m_top = %d, max = %d\n", m_top, m_max);
        assert(m_top+cnt <= m_max);
    #endif
        u32 old = atomicAdd(&m_top, cnt);
        return m_elements + old;
    }
};


struct bmTreeNode
{
    union
    {
        bmTreeNode* m_left;    
        bmFace** m_faces;
    };
    bmTreeNode* m_right;
    u32 m_faceInsertIdx;

    __device__ void init();
    __device__ void split(bmStore<bmTreeNode>* store);
    __device__ void insertFace(bmStore<bmFace>* faceStore, bmStore<bmFace*>* faceGroupStore, u32 meshIdx, uint3 faceIdx, bmMaterial* mat);
};


struct bmStackNode
{
    vec3 m_min, m_max;
    bmTreeNode* m_node;
    u32 m_splitAxis;
    u32 m_depth;

    __device__ void init(bmTreeNode* node, const vec3& bMin, const vec3& bMax, u32 splitAxis, u32 depth);
    __device__ bool intersect(const vec3& triMin, const vec3& triMax);
     __device__ void splitOrAdd(bmStackNode* left, bmStackNode* right, bmStore<bmTreeNode>* nodeStore);
};


#endif