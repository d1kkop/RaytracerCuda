#pragma once

#include "Beam.h"
#include "glm/mat4x4.hpp"
#include <cuda_runtime.h>


namespace Beam
{
    struct StaticMeshData
    {
        /* All Device ptrs */
        float* m_vertexData[VERTEX_DATA_COUNT];
        u32  m_vertexDataSizes[VERTEX_DATA_COUNT];
        u32* m_indices;
        u32  m_numVertices;
        u32  m_numIndices;
    };

    struct bmMaterial
    {
        int texture;
    };

    struct bmFace
    {
        uint4 m_index; // x,y,z indices, w meshIdx
        bmMaterial* m_material;
    };

    struct bmRayFaceHitResult
    {
        bmFace* face;
        float dist, u, v;
    };
    
    using bmHitCallback = void (*)( u32, const bmRayFaceHitResult*, const StaticMeshData*, u32* buffer );
}
