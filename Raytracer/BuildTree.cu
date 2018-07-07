#include "BuildTree.cuh"

// -------- bmFace ---------------------------------------------------------------------------------------------------------------

__device__ float bmFace::intersect(const vec3& eye, const vec3& dir, const StaticMeshData* meshDataPtrs, u32 numMeshes, float& u, float& v)
{
    assert( meshDataPtrs );
    assert( m_index.w < numMeshes );
    const StaticMeshData* mesh = &meshDataPtrs[m_index.w];
    uint4 idx = m_index;
    float* vp  = mesh->m_vertexData[ VERTEX_DATA_POSITION ];
    u32 vpSize = mesh->m_vertexDataSizes[ VERTEX_DATA_POSITION ];
    assert( vpSize >= 3 );
    vec3 v1 = *(vec3*)(vp + vpSize*idx.x);
    vec3 v2 = *(vec3*)(vp + vpSize*idx.y);
    vec3 v3 = *(vec3*)(vp + vpSize*idx.z);
    return bmTriIntersect( eye, dir, v1, v2, v3, u, v );
}

// -------- bmTreeNode -----------------------------------------------------------------------------------------------------------

__device__ void bmTreeNode::init()
{
    m_left = m_right = nullptr;
    m_faceRoot  = nullptr;
    m_faceTail  = nullptr;
    m_syncSplit = 0;
    m_syncInsertFace = 0;
    m_ready = true;
}

__device__ void bmTreeNode::split(bmStore<bmTreeNode>* store)
{
    if ( 0 == atomicAdd(&m_syncSplit, 1) )
    {
        m_left  = store->getNew();
        m_right = store->getNew();
        m_left->init();
        m_right->init();
        //__threadfence(); __threadfence();
    }
    __threadfence();
    //__syncthreads();
    while ( !(m_right && m_right->m_ready) );
    assert( m_left );
    assert( m_right );
}

__device__ void bmTreeNode::insertFace( bmStore<bmFace>* faceStore, u32 meshIdx, uint3 faceIdx, bmMaterial* mat )
{
    bmFace* face = faceStore->getNew();
    face->m_index.x = faceIdx.x;
    face->m_index.y = faceIdx.y;
    face->m_index.z = faceIdx.z;
    face->m_index.w = meshIdx;
    face->m_material = mat;
    face->m_next = nullptr;
    while ( atomicCAS(&m_syncInsertFace, 0, 1) )
    {
        if ( !m_faceTail )
        {
            m_faceRoot = face;
            m_faceTail = face;
        }
        else
        {
            m_faceTail->m_next = face;
            m_faceTail = m_faceTail->m_next;
        }
    }
    atomicExch( &m_syncInsertFace, 0 );
}


// -------- bmStackNode -----------------------------------------------------------------------------------------------------------


__device__ void bmStackNode_init(bmStackNode* st, bmTreeNode* node, const vec3& bMin, const vec3& bMax, u32 splitAxis, u32 depth)
{
    st->init( node, bMin, bMax, splitAxis, depth );
}

__device__ void bmStackNode::init( bmTreeNode* node, const vec3& bMin, const vec3& bMax, u32 splitAxis, u32 depth )
//__device__ void bmStackNode::init( bmTreeNode* node, vec3 bMin, vec3 bMax, u32 splitAxis, u32 depth )
{
    m_min  = bMin;
    m_max  = bMax;
    m_node = node;
    m_splitAxis = splitAxis;
    m_depth = depth;
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
    float s = .5f*(m_max[m_splitAxis]+m_min[m_splitAxis]);
    switch ( m_splitAxis )
    {
    case 0:
        left->init ( m_node->m_left, m_min, vec3(s, m_max.y, m_max.z), 1, m_depth+1 );
        right->init( m_node->m_right, vec3(s, m_min.y, m_min.z), m_max, 1, m_depth+1 );
        break;
    
    case 1:
        left->init ( m_node->m_left, m_min, vec3(m_max.x, s, m_max.z), 2, m_depth+1 );
        right->init( m_node->m_right, vec3(m_min.x, s, m_min.z), m_max, 2, m_depth+1 );
        break;

    case 2:
        left->init ( m_node->m_left, m_min, vec3(m_max.x, m_max.y, s), 0, m_depth+1 );
        right->init( m_node->m_right, vec3(m_min.x, m_min.y, s), m_max, 0, m_depth+1 );
        break;

    default:
        assert(false);
        break;
    }
}


// -------- bmResetSceneKernel -----------------------------------------------------------------------------------------------------------

__global__ void bmResetSceneKernel( bmTreeNode* rootNode,
                                    bmStore<bmFace>* faceStore, bmStore<bmTreeNode>* nodeStore, 
                                    bmFace* faces, bmTreeNode* nodes,
                                    u32 maxFaces, u32 maxNodes )
{
    rootNode->m_left  = nullptr;
    rootNode->m_right = nullptr;
    rootNode->m_faceRoot  = nullptr;
    rootNode->m_faceTail  = nullptr;
    rootNode->m_syncSplit = 0;
    rootNode->m_syncInsertFace = 0;
    rootNode->m_ready = false;

    faceStore->m_elements = faces;
    faceStore->m_top = 0;
    faceStore->m_max = maxFaces;

    nodeStore->m_elements = nodes;
    nodeStore->m_top = 0;
    nodeStore->m_max = maxNodes;
}


extern "C"
void bmResetScene( void* rootNode, void* faceStore, void* nodeStore, 
                   void* faces, void* nodes, 
                   u32 maxFaces, u32 maxNodes )
{
    bmTreeNode* t_rootNode           = (bmTreeNode*) rootNode;
    bmStore<bmFace>* t_faceStore     = (bmStore<bmFace>*) faceStore;
    bmStore<bmTreeNode>* t_nodeStore = (bmStore<bmTreeNode>*) nodeStore;
    bmFace* t_faces                  = (bmFace*) faces;
    bmTreeNode* t_nodes              = (bmTreeNode*) nodes;

    bmResetSceneKernel<<< 1, 1 >>>
    ( 
        t_rootNode, t_faceStore, t_nodeStore, t_faces, t_nodes,
        maxFaces, maxNodes
    );
}


// -------- bmInserTriangleInTree -----------------------------------------------------------------------------------------------------------

__global__ void bmInsertTriangleInTree( const vec3* vertices, const uint3* faces, 
                                        u32 numFaces, u32 meshIdx,
                                        vec3 bMin, vec3 bMax, 
                                        bmTreeNode* root, bmStore<bmFace>* faceStore,
                                        bmStore<bmTreeNode>* nodeStore, bmMaterial* mat )
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

  //  __shared__ bmStackNode stack[BUILD_TREE_THREADS][BUILD_TREE_MAX_DEPTH];
    bmStackNode stack[BUILD_TREE_MAX_DEPTH];
    bmStackNode* st = stack;// stack[tIdx];
    st[0].init( root, bMin, bMax, 0, 0 );
    i32 top = 0;
    
    do
    {
        bmStackNode* stNode = &st[top--];
        assert( stNode );
        assert( stNode->m_node );
        if ( stNode->intersect( triMin, triMax ) )
        {
            if ( stNode->m_depth == BUILD_TREE_MAX_DEPTH-1 )
            {
                stNode->m_node->insertFace( faceStore, meshIdx, id, mat );
                break;
            }
            else
            {
                assert( top+2 <= BUILD_TREE_MAX_DEPTH );
                bmStackNode* l = &st[++top];
                bmStackNode* r = &st[++top];
                stNode->splitOrAdd( r, l, nodeStore );
                assert( l->m_node );
                assert( r->m_node );
            }
        }
    } while ( top >= 0 );

}


extern "C"
void bmInsertMeshInTree( const vec3* vertices, const u32* indices, u32 numIndices, u32 meshIdx,
                         vec3 bMin, vec3 bMax,
                         void* treeRootNode, void* faceStore, void* nodeStore, void* material )
{
    u32 numFaces = numIndices / 3;
    assert( numIndices%3 == 0 );

    dim3 blocks ( (numFaces+BUILD_TREE_THREADS-1)/BUILD_TREE_THREADS );
    dim3 threads( BUILD_TREE_THREADS );

    bmTreeNode* t_rootNode           = (bmTreeNode*) treeRootNode;
    bmStore<bmFace>* t_faceStore     = (bmStore<bmFace>*) faceStore;
    bmStore<bmTreeNode>* t_nodeStore = (bmStore<bmTreeNode>*) nodeStore;
    bmMaterial* t_material           = (bmMaterial*)material;

    bmInsertTriangleInTree<<< blocks, threads >>>
    ( 
        vertices, (uint3*)indices, numFaces, meshIdx,
        bMin, bMax, 
        t_rootNode, t_faceStore, t_nodeStore, t_material
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
    st->init( root, bMin, bMax, 0, BUILD_TREE_MAX_DEPTH );
    i32 top  = 0;
    float dClosest = FLT_MAX;
//    float tU, tV;    

    buffer[i] = 255 << 8;
    return;

    do
    {
        st = &stack[top--];
        assert(st);
        assert(st->m_node);
        // check ray box intersect
        float boxDist = bmBoxRayIntersect(st->m_min, st->m_max, eye, invDir);
        if ( boxDist != FLT_MAX )
        {
            if ( !st->m_node->m_left ) // is leaf
            {
                // check intersections ray triangle list
                bmFace* face = st->m_node->m_faceRoot;
                while ( face )
                {
                    float u, v;
                    float d = face->intersect(eye, dir, meshDataPtrs, numMeshes, u, v);
                    if ( d < dClosest )
                    {
                        dClosest = d;
                  //      tU = u;
                  //      tV = v;
                    }
                    face = face->m_next;
                }
                if ( dClosest != FLT_MAX )
                {
                    break;
                }
            }
            else
            {
                u32 axis   = st->m_splitAxis;
                u32 naxis  = (axis+1)%3;
                u32 ndepth = st->m_depth+1;
                vec3 min   = st->m_min;
                vec3 max   = st->m_max;
                float s    = .5f*(max[axis]+min[axis]);
                vec3 p     = eye + boxDist*dir;
                if ( p[axis] < s ) // on left side
                {
                    float t = min[axis];
                    min[axis] = s;
                    st = &stack[++top];
                    st->init(st->m_node->m_right, min, max, naxis, ndepth);
                    min[axis] = t;
                    max[axis] = s;
                    st = &stack[++top];
                    st->init(st->m_node->m_left, min, max, naxis, ndepth);
                }
                else
                {
                    float t = max[axis];
                    max[axis] = s;
                    st = &stack[++top];
                    st->init(st->m_node->m_left, min, max, naxis, ndepth);
                    min[axis] = s;
                    max[axis] = t;
                    st = &stack[++top];
                    st->init(st->m_node->m_right, min, max, naxis, ndepth);
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
    u32 bmGetMaterialSize() { return sizeof(bmMaterial); }
    u32 bmGetNodeSize() { return sizeof(bmTreeNode); }
    u32 bmGetFaceStoreSize() { return sizeof(bmStore<bmFace>); }
    u32 bmGetNodeStoreSize() { return sizeof(bmStore<bmTreeNode>); }
}