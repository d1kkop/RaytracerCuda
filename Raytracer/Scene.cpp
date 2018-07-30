#include "Scene.h"
#include "SceneTree.h"
#include "SceneHash.h"
#include "SceneProgressive.h"
#include "Mesh.h"
#include "SharedTypes.h"
#include "DeviceBuffer.h"
#include "RenderTarget.h"
#include "BuildTree.cuh"
#include <algorithm>
#include <cassert>
#include <iostream>
using namespace std;
using namespace glm;
using namespace Beam;


namespace Beam
{
    sptr<IScene> IScene::create()
    {
    #if TREE_TYPE==TREE
        return make_shared<SceneTree>();
    #elif TREE_TYPE==HASH
        return make_shared<SceneHash>();
    #else TREE_TYPE==PROGRESSIVE
        return make_shared<SceneProgressive>();
    #endif
        return nullptr;
    }

    Scene::Scene():
        m_mustUpdateMeshPtrs(false)
    {
    }

    void Scene::addMesh(const sptr<IMesh>& mesh)
    {
        m_staticMeshes.emplace_back( static_pointer_cast<Mesh>(mesh) );
        m_mustUpdateMeshPtrs = true;
    }

    void Scene::removeMesh(const IMesh& mesh)
    {
        for ( auto it=m_staticMeshes.begin(); it!=m_staticMeshes.end(); it++ )
        {
            if ( &mesh == it->get() )
            {
                m_staticMeshes.erase(it);
                break;
            }
        }
    #if _DEBUG
        assert( find_if(m_staticMeshes.begin(), m_staticMeshes.end(), [&](auto& m) { return m.get()==&mesh; }) == m_staticMeshes.end() );
    #endif
    }

    void Scene::updateMeshPtrs()
    {
        if (!m_mustUpdateMeshPtrs) return;
        StaticMeshData* sceneMeshDataPtrs = new StaticMeshData[m_staticMeshes.size()];
        for (u32 i = 0; i < m_staticMeshes.size() ; i++)
        {
        	auto& meshDataPtr = sceneMeshDataPtrs[i];
            auto& staticMesh  = m_staticMeshes[i];
            meshDataPtr.m_indices = staticMesh->indices()->ptr<u32>();
            for ( u32 i=0; i<VERTEX_DATA_COUNT; ++i )
            {
                meshDataPtr.m_vertexData[i]         = staticMesh->vertexData(i).get() ? staticMesh->vertexData(i)->ptr<float>() : nullptr;
                meshDataPtr.m_vertexDataSizes[i]    = staticMesh->vertexDataSize(i);
            }
            meshDataPtr.m_numIndices  = staticMesh->numIndices();
            meshDataPtr.m_numVertices = staticMesh->numVertices();
        }
        m_staticMeshPtrs = make_shared<DeviceBuffer>( u32(m_staticMeshes.size() * sizeof(StaticMeshData)) );
        m_staticMeshPtrs->copyFrom(sceneMeshDataPtrs, true);
        delete [] sceneMeshDataPtrs;
        m_mustUpdateMeshPtrs = false;
    }

    u32 Scene::march(const glm::vec3& eye, const glm::mat3& orient, const sptr<DeviceBuffer>& rays, u32 raysWide, u32 raysHigh)
    {
        /* Get global render target */
        auto rt = RenderTarget::get();
        if ( !rt )
        {
            return ERROR_NO_RENDER_TARGET;
        }

        /* Ensure RT and Cam have same dimensions. */
        if ( rt->width() != raysWide || rt->height() != raysHigh )
        {
            return ERROR_RT_CAM_MISMATCH;
        }

        return ERROR_ALL_FINE;
    }



}