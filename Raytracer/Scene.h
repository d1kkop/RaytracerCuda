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

        const sptr<DeviceBuffer>& staticMeshPtrs() const { return m_staticMeshPtrs; }

    #if TREE
        const sptr<DeviceBuffer>& rootNode() const { return m_rootNode; }
        const glm::vec3& _min() const { return m_min; }
        const glm::vec3& _max() const { return m_max; }
    #else
        const sptr<DeviceBuffer>& cells() const { return m_cells; }
    #endif

    private:
        void addMeshToSceneOnGPU( const Mesh& mesh, u32 meshIdx );

        Array<sptr<Mesh>>  m_staticMeshes;
        sptr<DeviceBuffer> m_staticMeshPtrs;
        bool m_mustUpdateMeshPtrs;

    #if TREE
        glm::vec3 m_min, m_max;
        sptr<DeviceBuffer> m_rootNode;
        sptr<DeviceBuffer> m_nodeStore;
        sptr<DeviceBuffer> m_faceStore;
        sptr<DeviceBuffer> m_faceGroupStore;
        sptr<DeviceBuffer> m_facesBuffer;
        sptr<DeviceBuffer> m_facePtrsBuffer;
        sptr<DeviceBuffer> m_nodesBuffer;
    #else
        sptr<DeviceBuffer> m_cells;
    #endif
    };

}