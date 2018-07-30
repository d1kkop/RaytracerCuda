#pragma once
#include "Scene.h"

#if TREE_TYPE==TREE

namespace Beam
{
    class DeviceBuffer;

    class SceneTree: public Scene
    {
    public:
        SceneTree();
        void updateGPUScene() override;
        const glm::vec3& _min() const { return m_min; }
        const glm::vec3& _max() const { return m_max; }
        u32 march(const glm::vec3& eye, const glm::mat3& orient, const sptr<DeviceBuffer>& rays, u32 raysWide, u32 raysHigh) override;

    private:
        void addMeshToSceneOnGPU(const Mesh& mesh, u32 meshIdx);

        glm::vec3 m_min, m_max;
        sptr<DeviceBuffer> m_rootNode;
        sptr<DeviceBuffer> m_nodeStore;
        sptr<DeviceBuffer> m_faceStore;
        sptr<DeviceBuffer> m_faceGroupStore;
        sptr<DeviceBuffer> m_facesBuffer;
        sptr<DeviceBuffer> m_facePtrsBuffer;
        sptr<DeviceBuffer> m_nodesBuffer;
    };
}

#endif