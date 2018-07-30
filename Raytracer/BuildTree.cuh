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

#endif