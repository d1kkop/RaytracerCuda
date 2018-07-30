#include "SceneHash.h"

#if TREE_TYPE==HASH

#include "Mesh.h"
#include "DeviceBuffer.h"
#include "RenderTarget.h"
using namespace std;

extern "C"
{
    void bmResetSpace( void* cells );
    void bmInsertMeshInSpace(const vec3* vertices,
                             const u32* indices, u32 numIndices,
                             void* material, u32 meshIdx,
                             void* cells);
    void bmMarchSpace(const vec3* initialRays,
                      u32* buffer, u32 pitch, u32 width, u32 height,
                      const vec3& eye, const mat3& orient,
                      const StaticMeshData* meshData,
                      void* cells);
    u32 bmGetCellSize();
    u32 bmGetMaxHashElements();
}


namespace Beam
{
    SceneHash::SceneHash()
    {
        m_cells = make_shared<DeviceBuffer>(bmGetCellSize()*bmGetMaxHashElements());
        float cMb  = (float)m_cells->size()/1024/1024;
        cout << "Cells memory: " << cMb << "mb." << endl;
        cout << "Total: " << cMb << "mb." << endl;
    }

    void SceneHash::updateGPUScene()
    {
        updateMeshPtrs();
        bmResetSpace( m_cells->ptr<void>() );
        for ( u32 i=0; i<m_staticMeshes.size(); i++ )
        {
            addMeshToSceneOnGPU(*m_staticMeshes[i], i );
        }
    }

    void SceneHash::addMeshToSceneOnGPU(const Mesh& mesh, u32 meshIdx)
    {
        bmInsertMeshInSpace( 
            mesh.vertexData(VERTEX_DATA_POSITION)->ptr<const vec3>(),
            mesh.indices()->ptr<u32>(),
            mesh.numIndices(),
            nullptr,
            meshIdx,
            m_cells->ptr<void>());
    }

    u32 SceneHash::march(const glm::vec3& eye, const glm::mat3& orient, const sptr<DeviceBuffer>& rays, u32 raysWide, u32 raysHigh)
    {
        u32 err = Scene::march( eye, orient, rays, raysWide, raysHigh );
        if ( err != ERROR_ALL_FINE ) return err;
        auto rt = RenderTarget::get();
        bmMarchSpace(
            rays->ptr<const vec3>(),
            rt->buffer<u32>(),
            rt->pitch(),
            rt->width(),
            rt->height(),
            eye, orient,
            m_staticMeshPtrs->ptr<const StaticMeshData>(),
            m_cells->ptr<void>()
        );
        return ERROR_ALL_FINE;
    }
}

#endif