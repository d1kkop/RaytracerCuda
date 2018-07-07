#pragma once
#ifndef BUILD_TREE_CUH
#define BUILD_TREE_CUH
#include "CudaComon.cuh"
#include "SharedTypes.h"
#include <cassert>
using namespace std;
using namespace Beam;


#define BUILD_TREE_THREADS 128
#define BUILD_TREE_MAX_DEPTH 12
#define MARCH_THREADS 256

struct bmMaterial
{
    int texture;
};


struct bmFace
{
    uint4 m_index; // x,y,z indices, w meshIdx
    bmFace* m_next;
    bmMaterial* m_material;
    __device__ float intersect(const vec3& eye, const vec3& dir, const StaticMeshData* meshDataPtrs, u32 numMeshes, float& u, float& v);
};


template <typename T>
struct bmStore
{
    T* m_elements;
    u32 m_top;
    u32 m_max;

    __forceinline__ __device__ T* getNew()
    {
    #if _DEBUG
        if ( m_top >= m_max )
            printf("m_top = %d, max = %d\n", m_top, m_max);
        assert(m_top < m_max);
    #endif
        u32 old = atomicAdd(&m_top, 1);
        return m_elements + old;
    }
};


struct bmTreeNode
{
    bmTreeNode* m_left;
    bmTreeNode* m_right;
    bmFace* m_faceRoot;
    bmFace* m_faceTail;
    u32 m_syncSplit, m_syncInsertFace;
    bool m_ready;

    __device__ void init();
    __device__ void split(bmStore<bmTreeNode>* store);
    __device__ void insertFace(bmStore<bmFace>* faceStore, u32 meshIdx, uint3 faceIdx, bmMaterial* mat);
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