#include "SceneProgressive.h"

#if TREE_TYPE==PROGRESSIVE

#include "Mesh.h"
#include "DeviceBuffer.h"



extern "C"
{
}


namespace Beam
{
    SceneProgressive::SceneProgressive()
    {
    }

    void SceneProgressive::updateGPUScene()
    {
        updateMeshPtrs();

    }

    void SceneProgressive::addMeshToSceneOnGPU(const Mesh& mesh, u32 meshIdx)
    {
  
    }

    u32 SceneProgressive::march(const glm::vec3& eye, const glm::mat3& orient, const sptr<DeviceBuffer>& rays, u32 raysWide, u32 raysHigh)
    {
        u32 err = Scene::march( eye, orient, rays, raysWide, raysHigh );
        if ( err != ERROR_ALL_FINE ) return err;

        return ERROR_ALL_FINE;
    }
}

#endif