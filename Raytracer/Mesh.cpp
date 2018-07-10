#include "Mesh.h"
#include "Scene.h"
#include "DeviceBuffer.h"
#include "Util.h"
#include <cassert>
using namespace std;

namespace Beam
{
    // ------ IMesh ----------------------------------------------------------------------------------------

    sptr<IMesh> IMesh::create()
    {
        return make_shared<Mesh>();
    }

    // ------ Mesh ----------------------------------------------------------------------------------------

    Mesh::Mesh():
        m_numVertices(0),
        m_numIndices(0)
    {
        for (u32 & vds : m_vertexDataSizes) vds = 0;
    }

    Mesh::~Mesh()
    {
    }

    u32 Mesh::setVertexData(const float* vertexData, u32 numVertices, u32 numComponents, u32 slotId, bool asyncCopy)
    {
        if ( !vertexData || numVertices==0 || slotId >= VERTEX_DATA_COUNT || numComponents > 4 ||
             (m_numVertices!=0 && m_numVertices!=numVertices) ||
             (slotId == VERTEX_DATA_POSITION && numComponents != 3 )
            )
        {
            return ERROR_INVALID_PARAMETER;
        }
        m_vertexData[slotId] = make_shared<DeviceBuffer>((u32)sizeof(float)*numComponents*numVertices);
        if ( !m_vertexData[slotId] ) return ERROR_GPU_ALLOC_FAIL;
        m_vertexData[slotId]->copyFrom( vertexData, !asyncCopy );
        m_vertexDataSizes[slotId] = numComponents;
        return ERROR_ALL_FINE;
    }

    u32 Mesh::setIndices(const u32* indices, u32 numIndices, bool asyncCopy)
    {
        if ( !indices || (numIndices%3)!=0 ) return ERROR_INVALID_PARAMETER;
        m_numIndices = numIndices;
        m_indices = make_shared<DeviceBuffer>(m_numIndices*(u32)sizeof(u32));
        if ( !m_indices ) return ERROR_GPU_ALLOC_FAIL;
        m_indices->copyFrom(indices, !asyncCopy);
        return ERROR_ALL_FINE;
    }

    wptr<IScene> Mesh::scene() const 
    { 
        return m_scene; 
    }
}
