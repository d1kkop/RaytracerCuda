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
        void updateGPUScene() override;

        const glm::vec3& min() const { return m_min; }
        const glm::vec3& max() const { return m_max; }

        const sptr<DeviceBuffer>& rootNode() const { return m_rootNode; }
        const sptr<DeviceBuffer>& staticMeshPtrs() const { return m_staticMeshPtrs; }
        

    private:
        void addMeshToSceneOnGPU( const Mesh& mesh, u32 meshIdx );

        glm::vec3 m_min, m_max;
        Array<sptr<Mesh>> m_staticMeshes;
        sptr<DeviceBuffer> m_staticMeshPtrs;
        sptr<DeviceBuffer> m_rootNode;
        sptr<DeviceBuffer> m_nodeStore;
        sptr<DeviceBuffer> m_faceStore;
        sptr<DeviceBuffer> m_facesBuffer;
        sptr<DeviceBuffer> m_nodesBuffer;
        bool m_mustUpdateMeshPtrs;
    };

}