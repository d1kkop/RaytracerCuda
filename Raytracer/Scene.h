#pragma once
#include "Beam.h"
#include "glm/glm.hpp"

namespace Beam
{
    class Mesh;
    class DeviceBuffer;


    class Scene: public IScene
    {
    public:
        Scene();

        void addMesh( const sptr<IMesh>& mesh ) override;
        void removeMesh( const IMesh& mesh ) override;
        void updateMeshPtrs();

        virtual u32 march(const glm::vec3& eye, const glm::mat3& orient, const sptr<DeviceBuffer>& rays, u32 raysWide, u32 raysHigh);

    protected:
        Array<sptr<Mesh>>  m_staticMeshes;
        sptr<DeviceBuffer> m_staticMeshPtrs;
        bool m_mustUpdateMeshPtrs;
    };

}