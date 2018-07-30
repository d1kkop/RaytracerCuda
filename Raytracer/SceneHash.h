#pragma once
#include "Scene.h"

#if TREE_TYPE==HASH

namespace Beam
{
    class SceneHash: public Scene
    {
    public:
        SceneHash();
        void updateGPUScene() override;
        const sptr<DeviceBuffer>& cells() const { return m_cells; }
        u32 march(const glm::vec3& eye, const glm::mat3& orient, const sptr<DeviceBuffer>& rays, u32 raysWide, u32 raysHigh) override;

    private:
        void addMeshToSceneOnGPU( const Mesh& mesh, u32 meshIdx );

        sptr<DeviceBuffer> m_cells;
    };

}

#endif