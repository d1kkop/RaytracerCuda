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
    };

}

#endif