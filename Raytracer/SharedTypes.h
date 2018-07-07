#pragma once

#include "Beam.h"
#include "glm/mat4x4.hpp"
#include <cuda_runtime.h>


namespace Beam
{
    __align__(16)
    struct StaticMeshData
    {
        /* All Device ptrs */
        float* m_vertexData[VERTEX_DATA_COUNT];
        u32  m_vertexDataSizes[VERTEX_DATA_COUNT];
        u32* m_indices;
        u32  m_numVertices;
        u32  m_numIndices;
    };

    __align__(16)
    struct DynamicMeshData: public StaticMeshData
    {
        glm::mat4 m_transform;
    };
}
