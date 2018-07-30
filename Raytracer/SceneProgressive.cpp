#include "SceneProgressive.h"

#if TREE_TYPE==PROGRESSIVE

#include "Mesh.h"
#include "DeviceBuffer.h"
using namespace Beam;
using namespace std;

// Names: ReyLax


extern "C"
{
    u32 bmGetStoreSize();
    u32 bmGetFaceClusterSize();
    u32 bmGetBvhNodeSize();
    u32 bmGetRaySize();
    u32 bmGetRayBoxSize();
    u32 bmGetRayFaceSize();
    u32 bmRayFaceHitClusterSize();
}


namespace Beam
{
    SceneProgressive::SceneProgressive()
    {
        u32 numRays = 0;
        m_rayBoxStore   = make_shared<DeviceBuffer>( bmGetStoreSize() );
        m_rayBoxBuffer  = make_shared<DeviceBuffer>( bmGetRayBoxSize() * numRays * 256 );
        m_leafStore     = make_shared<DeviceBuffer>( bmGetStoreSize() );
        m_leafBuffer    = make_shared<DeviceBuffer>( (u32)sizeof(u32) * numRays * 16 );
        m_rayFaceStore      = make_shared<DeviceBuffer>( bmGetStoreSize());
        m_rayFaceBuffer     = make_shared<DeviceBuffer>( bmGetRayFaceSize() * numRays * 16 );
        m_faceClusters      = make_shared<DeviceBuffer>( bmGetFaceClusterSize() * numRays );
        m_hitResultClusters = make_shared<DeviceBuffer>( bmRayFaceHitClusterSize() * numRays );

        float rbSize =      (float)m_rayFaceStore->size()/1024/1024;
        float rbBuffSize =  (float)m_rayBoxBuffer->size()/1024/1024;
        float lfSize =      (float)m_leafStore->size()/1024/1024;
        float lfBuffSize =  (float)m_leafBuffer->size()/1024/1024;
        float rfaceSize =   (float)m_rayFaceStore->size()/1024/1024;
        float rfaceBuffSize = (float)m_rayFaceBuffer->size()/1024/1024;
        float fClusters =   (float)m_faceClusters->size()/1024/1024;
        float hClusters =   (float)m_hitResultClusters->size()/1024/1024;

        printf("RayBox store:\t%fmb\n", rbSize);
        printf("RayBox buffer:\t%fmb\n", rbBuffSize);
        printf("Leaf store:\t%fmb\n", lfSize);
        printf("Leaf buffer:\t%fmb\n", lfBuffSize);
        printf("RayFace store:\t%fmb\n", rfaceSize);
        printf("RayFace buffer:\t%fmb\n", rfaceBuffSize);
        printf("Face clusters:\t%fmb\n", fClusters);
        printf("Hit clusters:\t%fmb\n", hClusters);

        printf("Total: %f\n", rbSize+rfaceBuffSize+lfSize+lfBuffSize+rfaceSize+rfaceBuffSize+fClusters+hClusters);
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