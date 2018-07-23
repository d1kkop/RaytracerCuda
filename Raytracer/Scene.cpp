#include "Scene.h"
#include "Mesh.h"
#include "SharedTypes.h"
#include "DeviceBuffer.h"
#include "BuildTree.cuh"
#include <algorithm>
#include <cassert>
#include <iostream>
using namespace std;
using namespace glm;
using namespace Beam;


extern "C"
{
    void bmResetScene(void* rootNode, void* faceStore,
                      void* faceGroupStore, void* nodeStore,
                      void* faces, void* facePtrs, void* nodes,
                      u32 maxFaces, u32 maxFacePtrs, u32 maxNodes);

    void bmInsertMeshInTree(const vec3* vertices, const u32* indices, u32 numIndices, u32 meshIdx,
                            vec3 bMin, vec3 bMax,
                            void* treeRootNode, void* faceStore, void* faceGroupStore, void* nodeStore, void* material);
    u32 bmGetFaceSize();
    u32 bmGetFacePtrSize();
    u32 bmGetMaterialSize();
    u32 bmGetNodeSize();
    u32 bmGetFaceStoreSize();
    u32 bmGetNodeStoreSize();
    u32 bmGetFaceGroupStoreSize();
}


namespace Beam
{
    constexpr u32 MaxNodes = (1<<16) * 16;               /* This is unrelated to max tree depth */
    constexpr u32 MaxFaces = (1<<16) * 64 * 4;

    // ------ IScene ----------------------------------------------------------------------------------------

    sptr<IScene> IScene::create()
    {
        return make_shared<Scene>();
    }

    // ------ Scene ----------------------------------------------------------------------------------------

    Scene::Scene():
        m_mustUpdateMeshPtrs(false),
        m_min(vec3(-30)),
        m_max(vec3(30))
    {
        u32 nodeSize    = MaxNodes * bmGetNodeSize();
        u32 faceSize    = MaxFaces * bmGetFaceSize();
        u32 facePtrSize = MaxFaces * bmGetFacePtrSize();
        assert( nodeSize && facePtrSize && faceSize );

        m_rootNode         = make_shared<DeviceBuffer>( bmGetNodeSize() );
        m_nodeStore        = make_shared<DeviceBuffer>( bmGetNodeStoreSize() );
        m_faceStore        = make_shared<DeviceBuffer>( bmGetFaceStoreSize() );
        m_faceGroupStore   = make_shared<DeviceBuffer>( bmGetFaceGroupStoreSize() );
        m_nodesBuffer      = make_shared<DeviceBuffer>( nodeSize );
        m_facesBuffer      = make_shared<DeviceBuffer>( faceSize );
        m_facePtrsBuffer   = make_shared<DeviceBuffer>( facePtrSize );

        float nMb  = (float)bmGetNodeSize()*MaxNodes/1024/1024;
        float fMb  = (float)bmGetFaceSize()*MaxFaces/1024/1024;
        float fpMb = (float)bmGetFacePtrSize()*MaxFaces/1024/1024;
        cout << "Nodes memory: " << nMb  << "mb." << endl;
        cout << "Faces memory: " << fMb  << "mb." << endl;
        cout << "FPtrs memory: " << fpMb << "mb." << endl;
        cout << "Total: " << nMb+fMb+fpMb << "mb." << endl;
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

    void Scene::updateGPUScene()
    {
        updateMeshPtrs();

        bmResetScene(
           m_rootNode->ptr<void>(),
           /* stores */
           m_faceStore->ptr<void>(),
           m_faceGroupStore->ptr<void>(),
           m_nodeStore->ptr<void>(),
            /* buffesr */
           m_facesBuffer->ptr<void>(),
           m_facePtrsBuffer->ptr<void>(),
           m_nodesBuffer->ptr<void>(),
            /* stores' _max elements */
           MaxFaces, MaxFaces, MaxNodes );
       
       for ( u32 i=0; i<m_staticMeshes.size(); i++ )
       {
           addMeshToSceneOnGPU(*m_staticMeshes[i], i );
       }

   //    cudaDeviceSynchronize();
    }

    void Scene::addMeshToSceneOnGPU(const Mesh& mesh, u32 meshIdx)
    {
        bmInsertMeshInTree( 
            mesh.vertexData( VERTEX_DATA_POSITION )->ptr<const vec3>(),
            mesh.indices()->ptr<u32>(), 
            mesh.numIndices(), 
            meshIdx,
            m_min, m_max, 
            m_rootNode->ptr<void>(), 
            m_faceStore->ptr<void>(),
            m_faceGroupStore->ptr<void>(),
            m_nodeStore->ptr<void>(),
            nullptr);
    }

}