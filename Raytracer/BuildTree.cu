#include "BuildTree.cuh"
#include "CudaHelp.h"

// -------- bmFace ---------------------------------------------------------------------------------------------------------------

__device__ float bmFace::intersect(const vec3& eye, const vec3& dir, const StaticMeshData* meshDataPtrs, u32 numMeshes, float& u, float& v)
{
    assert( meshDataPtrs );
    assert( m_index.w < numMeshes );
    const StaticMeshData* mesh = &meshDataPtrs[m_index.w];
    uint4 idx  = m_index;
    float* vp  = mesh->m_vertexData[ VERTEX_DATA_POSITION ];
    assert( mesh->m_vertexDataSizes[ VERTEX_DATA_POSITION ] == 3 );
    vec3 v1 = *(vec3*)(vp + 3*idx.x);
    vec3 v2 = *(vec3*)(vp + 3*idx.y);
    vec3 v3 = *(vec3*)(vp + 3*idx.z);
    return bmTriIntersect( eye, dir, v1, v2, v3, u, v );
}

// -------- bmTreeNode -----------------------------------------------------------------------------------------------------------

__device__ void bmTreeNode::init()
{
    m_left  = nullptr;
    m_right = nullptr;
    m_faceInsertIdx = 0;
}

__device__ void bmTreeNode::split(bmStore<bmTreeNode>* store)
{
    if ( !(m_left && m_right) )
    {
        // NOTE: If left and/or right are not assigned due to atomic compare & swap memory is LEAKDED.
        // However, the leaked memory is reclaimed every frame.
        bmTreeNode* leftAndRight = store->getNew(2);
        bool leftSwapped  = atomicCAS( (u64*)&m_left, (u64)0, (u64)leftAndRight ) == 0;
        bool rightSwapped = atomicCAS( (u64*)&m_right, (u64)0, (u64)(leftAndRight+1) ) == 0;
        if ( leftSwapped ) m_left->init();
        if ( rightSwapped) m_right->init();
    }
    __threadfence();
    assert( m_left );
    assert( m_right );
}

__device__ void bmTreeNode::insertFace( bmStore<bmFace>* faceStore, bmStore<bmFace*>* faceGroupStore, u32 meshIdx, uint3 faceIdx, bmMaterial* mat )
{
    bmFace* face  = faceStore->getNew();
    face->m_index = make_uint4(faceIdx.x, faceIdx.y, faceIdx.z, meshIdx);
    face->m_material = mat;

    assert( !m_right );
    if ( !m_faces ) // Multiple thread may pass this gate as it is not thread-safe.
    {
        // NOTE: If m_faces is already assigned, the allocated faceGroup is LEAKED. 
        // However, the memory is reclaimed the next frame.
        bmFace** faces = faceGroupStore->getNew( BUILD_TREE_MAX_DEPTH );
        atomicCAS( (u64*)&m_faces, (u64)0, (u64)faces ); 
    }
    __threadfence();
    assert( m_faces );

    u32 storeFidx = atomicAdd( &m_faceInsertIdx, 1 );
 //   assert( storeFidx < MAX_FACES_PER_BOX );
    if ( storeFidx < MAX_FACES_PER_BOX )
    {
      //  atomicExch( (u64)(m_faces+storeFidx), (u64)face );
        m_faces[storeFidx] = face;
    }
}


// -------- bmStackNode -----------------------------------------------------------------------------------------------------------



__device__ void bmStackNode::init( bmTreeNode* node, const vec3& bMin, const vec3& bMax, u32 splitAxis, u32 depth )
{
    m_min   = bMin;
    m_max   = bMax;
    m_node  = node;
    m_depth = depth;
    m_splitAxis = splitAxis;
}

__device__ bool bmStackNode::intersect(const vec3& triMin, const vec3& triMax)
{
    u32 misCount = 0;
    #pragma unroll
    for ( int i=0; i<3; ++i )
    {
        misCount += triMin[i] > m_max[i];
        misCount += triMax[i] < m_min[i];
    }
    return misCount == 0;
}

__device__ void bmStackNode::splitOrAdd( bmStackNode* left, bmStackNode* right, bmStore<bmTreeNode>* nodeStore )
{
    assert( m_node );
    assert( nodeStore );
    m_node->split( nodeStore );
    assert( m_node->m_left );
    assert( m_node->m_right );
    u32 ndepth = m_depth+1;
    float s = .5f*(m_max[m_splitAxis]+m_min[m_splitAxis]);
    switch ( m_splitAxis )
    {
    case 0:
        left->init ( m_node->m_left,  m_min, vec3(s, m_max.y, m_max.z), 1, ndepth );
        right->init( m_node->m_right, vec3(s, m_min.y, m_min.z), m_max, 1, ndepth );
        break;
    
    case 1:
        left->init ( m_node->m_left,  m_min, vec3(m_max.x, s, m_max.z), 2, ndepth );
        right->init( m_node->m_right, vec3(m_min.x, s, m_min.z), m_max, 2, ndepth );
        break;

    case 2:
        left->init ( m_node->m_left,  m_min, vec3(m_max.x, m_max.y, s), 0, ndepth );
        right->init( m_node->m_right, vec3(m_min.x, m_min.y, s), m_max, 0, ndepth );
        break;

    default:
        assert(false);
        break;
    }
}


// -------- bmResetSceneKernel -----------------------------------------------------------------------------------------------------------

__global__ void bmResetSceneKernel( bmTreeNode* rootNode,
                                    bmStore<bmFace>* faceStore, 
                                    bmStore<bmFace*>* faceGroupStore,
                                    bmStore<bmTreeNode>* nodeStore, 
                                    bmFace* faces, bmFace** facePtrs, bmTreeNode* nodes,
                                    u32 maxFaces, u32 maxFacePtrs, u32 maxNodes )
{
    rootNode->m_left  = nullptr;
    rootNode->m_right = nullptr;
    rootNode->m_faceInsertIdx = 0;

    faceStore->m_elements = faces;
    faceStore->m_top = 0;
    faceStore->m_max = maxFaces;

    faceGroupStore->m_elements = facePtrs;
    faceGroupStore->m_top = 0;
    faceGroupStore->m_max = maxFacePtrs;

    nodeStore->m_elements = nodes;
    nodeStore->m_top = 0;
    nodeStore->m_max = maxNodes;
}


extern "C"
void bmResetScene( void* rootNode, void* faceStore, void* faceGroupStore, void* nodeStore,
                   void* faces, void* facePtrs, void* nodes, 
                   u32 maxFaces, u32 maxFacePtrs, u32 maxNodes )
{
    assert( maxNodes >= 1<<BUILD_TREE_MAX_DEPTH );
    assert( maxFaces >= maxNodes * MAX_FACES_PER_BOX );

    bmTreeNode* t_rootNode             = (bmTreeNode*) rootNode;
    bmStore<bmFace>* t_faceStore       = (bmStore<bmFace>*) faceStore;
    bmStore<bmFace*>* t_faceGroupStore = (bmStore<bmFace*>*) faceGroupStore;
    bmStore<bmTreeNode>* t_nodeStore   = (bmStore<bmTreeNode>*) nodeStore;
    bmFace* t_faces                    = (bmFace*) faces;
    bmFace** t_facePtrs                = (bmFace**) facePtrs;
    bmTreeNode* t_nodes                = (bmTreeNode*) nodes;

    bmResetSceneKernel<<< 1, 1 >>>
    ( 
        t_rootNode, t_faceStore, t_faceGroupStore, t_nodeStore, 
        t_faces, t_facePtrs, t_nodes,
        maxFaces, maxFacePtrs, maxNodes
    );
}


// -------- bmInserTriangleInTree -----------------------------------------------------------------------------------------------------------

__global__ void bmInsertTriangleInTree( const vec3* vertices, const uint3* faces, 
                                        u32 numFaces, u32 meshIdx,
                                        vec3 bMin, vec3 bMax, 
                                        bmTreeNode* root, 
                                        bmStore<bmFace>* faceStore,
                                        bmStore<bmFace*>* faceGroupStore,
                                        bmStore<bmTreeNode>* nodeStore, 
                                        bmMaterial* mat )
{
    assert( vertices && faces && root && faceStore && nodeStore );
    assert( faceStore->m_max != 0 && nodeStore->m_max != 0 );

    // face index
    u32 tIdx = threadIdx.x;
    u32 fIdx = blockIdx.x * blockDim.x + tIdx;
    if ( fIdx >= numFaces ) return;
    uint3 id  = faces[fIdx];
    vec3 v[3] = { vertices[id.x], vertices[id.y], vertices[id.z] };

    // obtain aabb of triangle
    vec3 triMin, triMax;
    #pragma unroll
    for ( int i=0; i<3; ++i )
    {
        triMin[i] = min(v[0][i], min(v[1][i], v[2][i]));
        triMax[i] = max(v[0][i], max(v[1][i], v[2][i]));
    }

   // __shared__ bmStackNode stack_shared[BUILD_TREE_THREADS][BUILD_TREE_MAX_DEPTH];
   // bmStackNode* stack = stack_shared[tIdx];
    bmStackNode stack[BUILD_TREE_MAX_DEPTH];
    stack[0].init( root, bMin, bMax, 0, 0 );
    i32 top = 0;

    do
    {
        bmStackNode* stNode = &stack[top--];
        assert( stNode );
        assert( stNode->m_node );

        if ( stNode->intersect( triMin, triMax ) )
        {
           if ( stNode->m_depth == BUILD_TREE_MAX_DEPTH-1 )
           {
                stNode->m_node->insertFace( faceStore, faceGroupStore, meshIdx, id, mat );
          //      printf("Inserted..\n");
                break;
            }
            else
            {
          //      printf("Spliting...\n");
                assert( top+2 < BUILD_TREE_MAX_DEPTH );
                bmStackNode* l = &stack[++top];
                bmStackNode* r = &stack[++top];
                stNode->splitOrAdd( r, l, nodeStore );
                assert( l->m_node );
                assert( r->m_node );
            }
        }
    } while ( top >= 0 );
}


extern "C"
void bmInsertMeshInTree( const vec3* vertices, 
                         const u32* indices, u32 numIndices, u32 meshIdx,
                         vec3 bMin, vec3 bMax,
                         void* treeRootNode, 
                         void* faceStore, 
                         void* faceGroupStore,
                         void* nodeStore, 
                         void* material )
{
    u32 numFaces = numIndices / 3;
    assert( numIndices%3 == 0 );

    dim3 blocks ( (numFaces+BUILD_TREE_THREADS-1)/BUILD_TREE_THREADS );
    dim3 threads( BUILD_TREE_THREADS );

    bmTreeNode* t_rootNode             = (bmTreeNode*) treeRootNode;
    bmStore<bmFace>* t_faceStore       = (bmStore<bmFace>*) faceStore;
    bmStore<bmFace*>* t_faceGroupStore = (bmStore<bmFace*>*) faceGroupStore;
    bmStore<bmTreeNode>* t_nodeStore   = (bmStore<bmTreeNode>*) nodeStore;
    bmMaterial* t_material             = (bmMaterial*)material;

    bmInsertTriangleInTree<<< blocks, threads >>>
    ( 
        vertices, (uint3*)indices, numFaces, meshIdx,
        bMin, bMax, 
        t_rootNode, t_faceStore, t_faceGroupStore, 
        t_nodeStore, t_material
    );

}

// -------- bmMarchKernel -----------------------------------------------------------------------------------------------------------


__global__ void bmMarchKernel(bmTreeNode* root, vec3 bMin, vec3 bMax,
                              const vec3* initialRays, u32 numRays,
                              u32* buffer, vec3 eye, mat3 orient,
                              const StaticMeshData* meshDataPtrs, u32 numMeshes )
{
    assert(root && initialRays && buffer && meshDataPtrs);

    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    i = min(i, numRays-1);

    vec3 dir = initialRays[i];
    dir = orient*dir;
    vec3 invDir(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);

    bmStackNode stack[BUILD_TREE_MAX_DEPTH];
    bmStackNode* st = &stack[0];
    st->init( root, bMin, bMax, 0, 0 );
    i32 top  = 0;
    float dClosest = FLT_MAX;
//    float tU, tV;    

    do
    {
        assert( top >= 0 );
        st = &stack[top--];
        bmTreeNode* curNode = st->m_node;
        assert(st);
        assert(curNode);

        // check ray box intersect
        float boxDist = bmBoxRayIntersect(st->m_min, st->m_max, eye, invDir);
        if ( boxDist != FLT_MAX )
        {
            if ( !curNode->m_right ) // is leaf
            {
                if ( curNode->m_faces )
                {
                    // check intersections ray triangle list
                    u32 maxLoop = min(MAX_FACES_PER_BOX, curNode->m_faceInsertIdx);
                    for ( u32 i=0; i<maxLoop; i++ )
                    {
                        bmFace* face = curNode->m_faces[i];
                        float u, v;
                        float d = face->intersect(eye, dir, meshDataPtrs, numMeshes, u, v);
                        if ( d < dClosest )
                        {
                            dClosest = d;
                            //      tU = u;
                            //      tV = v;
                        }
                    }
                    if ( dClosest != FLT_MAX )
                    {
                        // if has hit, then this is always the most closest
                        break;
                    }
                }
            }
            else
            {
                assert( st );
                assert( top+2 < BUILD_TREE_MAX_DEPTH );
                assert( curNode->m_left );
                assert( curNode->m_right );
                u32 axis   = st->m_splitAxis;
                u32 naxis  = (axis+1)%3;
                vec3 min   = st->m_min;
                vec3 max   = st->m_max;
                float s    = .5f*(max[axis]+min[axis]);
                vec3 p     = eye + boxDist*dir;
                if ( p[axis] < s ) // on 'left' side
                {
                    // push right first
                    float t = min[axis];    // remember old min[axis]
                    min[axis] = s;          // overwrite
                    st = &stack[++top];     // push on stack
                    st->init(curNode->m_right, min, max, naxis, 0);
                    // push left now
                    min[axis] = t;  // restore min[axis]
                    max[axis] = s;  // set max[axis] for left
                    st = &stack[++top];
                    st->init(curNode->m_left, min, max, naxis, 0);
                }
                else // on 'right' side
                {
                    // push left first
                    float t = max[axis];    // remember old min[axis]
                    max[axis] = s;          // set max[axis] for left
                    st = &stack[++top];     // push on stack
                    st->init(curNode->m_left, min, max, naxis, 0);
                    // push right now
                    min[axis] = s;          // set min[axis] split value       
                    max[axis] = t;          // restore max[axis] value
                    st = &stack[++top];     // push
                    st->init(curNode->m_right, min, max, naxis, 0);
                }
            }
        }
    } while ( top >= 0 );
    
    if ( dClosest != FLT_MAX )
    {
        buffer[i] = 150<<16;
    }
    else
    {
        buffer[i] = 255<<8;
    }
}


extern "C"
void bmMarch(const void* root, const vec3& bMin, const vec3& bMax, 
             const vec3* initialRays,
             u32* buffer, u32 pitch, u32 width, u32 height,
             const vec3& eye, const mat3& orient,
             const StaticMeshData* meshData, u32 numMeshes)
{
    u32 numRays = width*height;
    dim3 blocks ( (numRays+MARCH_THREADS-1)/MARCH_THREADS );
    dim3 threads( MARCH_THREADS );
    bmMarchKernel<<< blocks, threads >>>
    ( 
        (bmTreeNode*)root,
        bMin, bMax,
        initialRays, numRays,
        buffer, eye, orient,
        meshData, numMeshes
    );
}


extern "C"
{
    u32 bmGetFaceSize() { return sizeof(bmFace); }
    u32 bmGetFacePtrSize() { return sizeof(bmFace*); }
    u32 bmGetMaterialSize() { return sizeof(bmMaterial); }
    u32 bmGetNodeSize() { return sizeof(bmTreeNode); }
    u32 bmGetFaceStoreSize() { return sizeof(bmStore<bmFace>); }
    u32 bmGetNodeStoreSize() { return sizeof(bmStore<bmTreeNode>); }
    u32 bmGetFaceGroupStoreSize() { return sizeof(bmStore<bmFace*>); }
}