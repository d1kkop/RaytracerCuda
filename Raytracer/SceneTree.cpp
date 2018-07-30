#include "SceneTree.h"

#if TREE_TYPE==TREE

#include "Mesh.h"
#include "DeviceBuffer.h"
#include "RenderTarget.h"
using namespace std;


extern "C"
{
    void bmResetScene(void* rootNode, void* faceStore,
                      void* faceGroupStore, void* nodeStore,
                      void* faces, void* facePtrs, void* nodes,
                      u32 maxFaces, u32 maxFacePtrs, u32 maxNodes);

    void bmInsertMeshInTree(const vec3* vertices, const u32* indices, u32 numIndices, u32 meshIdx,
                            vec3 bMin, vec3 bMax,
                            void* treeRootNode, void* faceStore, void* faceGroupStore, void* nodeStore, void* material);

    void bmMarch(const void* root, const vec3& bMin, const vec3& bMax,
                 const vec3* initialRays,
                 u32* buffer, u32 pitch, u32 width, u32 height,
                 const vec3& eye, const mat3& orient,
                 const StaticMeshData* meshData);

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
    constexpr u32 MaxNodes = (1<<16) * 64 * 4;               /* This is unrelated to max tree depth */
    constexpr u32 MaxFaces = (1<<16) * 64 * 4;

    SceneTree::SceneTree():
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

        float nMb  = (float)m_nodesBuffer->size()/1024/1024;
        float fMb  = (float)m_facesBuffer->size()/1024/1024;
        float fpMb = (float)m_facePtrsBuffer->size()/1024/1024;
        cout << "Nodes memory: " << nMb  << "mb." << endl;
        cout << "Faces memory: " << fMb  << "mb." << endl;
        cout << "FPtrs memory: " << fpMb << "mb." << endl;
        cout << "Total: " << nMb+fMb+fpMb << "mb." << endl;
    }

    void SceneTree::updateGPUScene()
    {
        updateMeshPtrs();

        bmResetScene(
           m_rootNode->ptr<void>(),
           /* stores */
           m_faceStore->ptr<void>(),
           m_faceGroupStore->ptr<void>(),
           m_nodeStore->ptr<void>(),
            /* buffers */
           m_facesBuffer->ptr<void>(),
           m_facePtrsBuffer->ptr<void>(),
           m_nodesBuffer->ptr<void>(),
            /* stores' _max elements */
           MaxFaces, MaxFaces, MaxNodes );
       
       for ( u32 i=0; i<m_staticMeshes.size(); i++ )
       {
           addMeshToSceneOnGPU(*m_staticMeshes[i], i );
       }
    }


    void SceneTree::addMeshToSceneOnGPU(const Mesh& mesh, u32 meshIdx)
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

    u32 SceneTree::march(const glm::vec3& eye, const glm::mat3& orient, const sptr<DeviceBuffer>& rays, u32 raysWide, u32 raysHigh)
    {
        u32 err = Scene::march(eye, orient, rays, raysWide, raysHigh);
        if ( err != ERROR_ALL_FINE ) return err;
        auto rt = RenderTarget::get();
        bmMarch
        (
            m_rootNode->ptr<void>(),
            m_min, m_max,
            rays->ptr<const vec3>(),
            rt->buffer<u32>(),
            rt->pitch(),
            rt->width(),
            rt->height(),
            eye, orient,
            m_staticMeshPtrs->ptr<const StaticMeshData>()
        );
        return ERROR_ALL_FINE;
    }
}

#endif