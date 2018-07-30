#pragma once
#include "Scene.h"

#if TREE_TYPE==PROGRESSIVE

namespace Beam
{
    class SceneProgressive: public Scene
    {
    public:
        SceneProgressive();
        void updateGPUScene() override;
        u32 march(const glm::vec3& eye, const glm::mat3& orient, const sptr<DeviceBuffer>& rays, u32 raysWide, u32 raysHigh) override;

    private:
        void addMeshToSceneOnGPU( const Mesh& mesh, u32 meshIdx );

        sptr<DeviceBuffer> m_rayBoxStore;
        sptr<DeviceBuffer> m_rayBoxBuffer;
        sptr<DeviceBuffer> m_leafStore;
        sptr<DeviceBuffer> m_leafBuffer;
        sptr<DeviceBuffer> m_rayFaceStore;
        sptr<DeviceBuffer> m_rayFaceBuffer;
        sptr<DeviceBuffer> m_faceClusters;
        sptr<DeviceBuffer> m_hitResultClusters;
    };

}

#endif