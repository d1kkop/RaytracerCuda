#pragma once

#include "Beam.h"

namespace Beam
{
    class Scene;
    class DeviceBuffer;

    class Mesh: public IMesh
    {
    public:
        Mesh();
        ~Mesh();

        u32 setVertexData(const float* data, u32 numVertices, u32 numComponents, u32 slotId, bool asyncCopy) override;
        u32 setIndices(const u32* indices, u32 numIndices, bool asyncCopy) override;
        wptr<IScene> scene() const override;

        u32 numVertices() const { return m_numVertices; }
        u32 numIndices() const  { return m_numIndices; }
        u32 vertexDataSize(u32 idx) const { return m_vertexDataSizes[idx]; }
        const sptr<DeviceBuffer>& indices() const { return m_indices; }
        const sptr<DeviceBuffer>& vertexData(u32 idx) const { return m_vertexData[idx]; }

    private:
        wptr<Scene> m_scene;
        u32 m_numVertices;
        u32 m_numIndices;
        sptr<DeviceBuffer> m_indices;
        sptr<DeviceBuffer> m_vertexData[VERTEX_DATA_COUNT];
        u32 m_vertexDataSizes[VERTEX_DATA_COUNT];

    };
}
