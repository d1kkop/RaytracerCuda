#include "BuildTree.cuh"
#include "CudaHelp.h"


// -------- bmFace ---------------------------------------------------------------------------------------------------------------

DEVICE float bmFace::intersect(const vec3& eye, const vec3& dir, const StaticMeshData* meshDataPtrs, u32 numMeshes, float& u, float& v)
{
    assert( meshDataPtrs );
    assert( m_index.w < numMeshes );
    const StaticMeshData* mesh = &meshDataPtrs[m_index.w];
    uint4 idx = m_index;
    vec3* vp  = (vec3*)mesh->m_vertexData[ VERTEX_DATA_POSITION ];
    assert( mesh->m_vertexDataSizes[ VERTEX_DATA_POSITION ] == 3 );
    return bmTriIntersect( eye, dir, vp[idx.x], vp[idx.y], vp[idx.z], u, v );
}

FDEVICE INLINE vec4 getData1(float* vd) { return vec4(vd[0], 0.f, 0.f, 0.f); }
FDEVICE INLINE vec4 getData2(float* vd) { return vec4(vd[0], vd[1], 0.f, 0.f); }
FDEVICE INLINE vec4 getData3(float* vd) { return vec4(vd[0], vd[1], vd[2], 0.f); }
FDEVICE INLINE vec4 getData4(float* vd) { return vec4(vd[0], vd[1], vd[2], vd[3]); }

using fGetData = vec4 (*)(float*);
CONSTANT fGetData c_getData[4] = 
{
    &getData1,
    &getData2,
    &getData3,
    &getData4
};

DEVICE vec4 bmFace::interpolate(float u, float v, const StaticMeshData* meshDataPtrs, u32 dataIdx)
{
    assert( meshDataPtrs );
    assert( dataIdx < VERTEX_DATA_COUNT );
    const StaticMeshData* mesh = &meshDataPtrs[m_index.w];
    uint4 idx = m_index;
    float* vd = mesh->m_vertexData[ dataIdx ];
    u32 dsize = mesh->m_vertexDataSizes[ dataIdx ];
    assert( dsize > 0 && dsize < 4 );
    float* vd1 = vd + idx.x*dsize;
    float* vd2 = vd + idx.y*dsize;
    float* vd3 = vd + idx.z*dsize;
    vec4 vu = c_getData[dsize]( vd1 ) * u;
    vec4 vv = c_getData[dsize]( vd2 ) * v;
    vec4 vw = c_getData[dsize]( vd3 ) * (1.f-(u+v));
    return vu + vv + vw;
}

// -------- bmTreeNode -----------------------------------------------------------------------------------------------------------

DEVICE void bmTreeNode::init()
{
    m_left  = nullptr;
    m_right = nullptr;
    m_faceInsertIdx = 0;
}

DEVICE void bmTreeNode::split(bmStore<bmTreeNode>* store)
{
    if ( !(m_left && m_right) )
    {
        // NOTE: If left and/or right are not assigned due to atomic compare & swap memory is LEAKDED.
        // However, the leaked memory is reclaimed every frame.
        bmTreeNode* leftAndRight = store->getNew(2);
        bool leftSwapped  = atomicCAS2<u64>( (u64*)&m_left, (u64)0, (u64)leftAndRight ) == 0;
        bool rightSwapped = atomicCAS2<u64>( (u64*)&m_right, (u64)0, (u64)(leftAndRight+1) ) == 0;
        if ( leftSwapped ) m_left->init();
        if ( rightSwapped) m_right->init();
    }
    THREAD_FENCE();
    assert( m_left );
    assert( m_right );
}

DEVICE void bmTreeNode::insertFace( bmStore<bmFace>* faceStore, bmStore<bmFace*>* faceGroupStore, u32 meshIdx, uint3 faceIdx, bmMaterial* mat )
{
    bmFace* face  = faceStore->getNew();
    face->m_index = make_uint4(faceIdx.x, faceIdx.y, faceIdx.z, meshIdx);
    face->m_material = mat;

    assert( !m_right );
    if ( !m_faces ) // Multiple thread may pass this gate as it is not thread-safe.
    {
        // NOTE: If m_faces is already assigned, the allocated faceGroup is LEAKED. 
        // However, the memory is reclaimed the next frame.
        bmFace** faces = faceGroupStore->getNew( MAX_FACES_PER_BOX );
        atomicCAS2<u64>( (u64*)&m_faces, (u64)0, (u64)faces ); 
    }
    THREAD_FENCE();
    assert( m_faces );

    u32 storeFidx = atomicAdd2<u32>( &m_faceInsertIdx, 1 );
  //  assert( storeFidx < MAX_FACES_PER_BOX );
    if ( storeFidx < MAX_FACES_PER_BOX )
    {
      //  atomicExch( (u64)(m_faces+storeFidx), (u64)face );
        m_faces[storeFidx] = face;
    }
  //  else printf("Num faces %d\n", storeFidx);
}


// -------- bmStackNode -----------------------------------------------------------------------------------------------------------



FDEVICE void bmStackNode::init( bmTreeNode* node, const vec3& bMin, const vec3& bMax, u32 splitAxis, u32 depth )
{
#if _DEBUG
    float dx = bMax.x-bMin.x;
    float dy = bMax.y-bMin.y;
    float dz = bMax.z-bMin.z; 
    if ( dx <= 0 || dy <= 0 || dz <= 0 )
    {
        printf("dx %.3f dy %.3f dz %.3f\n", dx, dy, dz);
        assert(false);
    }
#endif
    m_min   = bMin;
    m_max   = bMax;
    m_node  = node;
    m_depth = depth;
    m_splitAxis = splitAxis;
}

FDEVICE bool bmStackNode::intersect(const vec3& triMin, const vec3& triMax)
{
    bool b = 
        ( triMin[0] > m_max[0] ? false :
         ( triMin[1] > m_max[1] ? false :
          ( triMin[2] > m_max[2] ? false :
           ( triMax[0] < m_min[0] ? false :
            ( triMax[1] < m_min[1] ? false :
             ( triMax[2] < m_min[2] ? false : true ))))));
    return b;
}


// -------- bmResetSceneKernel -----------------------------------------------------------------------------------------------------------

GLOBAL void bmResetSceneKernel( bmTreeNode* rootNode,
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
    bmTreeNode* t_rootNode             = (bmTreeNode*) rootNode;
    bmStore<bmFace>* t_faceStore       = (bmStore<bmFace>*) faceStore;
    bmStore<bmFace*>* t_faceGroupStore = (bmStore<bmFace*>*) faceGroupStore;
    bmStore<bmTreeNode>* t_nodeStore   = (bmStore<bmTreeNode>*) nodeStore;
    bmFace* t_faces                    = (bmFace*) faces;
    bmFace** t_facePtrs                = (bmFace**) facePtrs;
    bmTreeNode* t_nodes                = (bmTreeNode*) nodes;

#if CUDA
    bmResetSceneKernel<<< 1, 1 >>>
    ( 
        t_rootNode, t_faceStore, t_faceGroupStore, t_nodeStore, 
        t_faces, t_facePtrs, t_nodes,
        maxFaces, maxFacePtrs, maxNodes
    );
#else
    bmResetSceneKernel 
    ( 
        t_rootNode, t_faceStore, t_faceGroupStore, t_nodeStore, 
        t_faces, t_facePtrs, t_nodes,
        maxFaces, maxFacePtrs, maxNodes
    );
#endif
}


// -------- bmInserTriangleInTree -----------------------------------------------------------------------------------------------------------

FDEVICE INLINE void getNewAABBAndSplitAxis0(float s, const vec3& bMin, const vec3& bMax, 
                                            vec3& nMaxL, vec3& nMinR, u32& axis)
{
    nMaxL = vec3(s, bMax.y, bMax.z);
    nMinR = vec3(s, bMin.y, bMin.z);
    axis  = 1;
}

FDEVICE INLINE void getNewAABBAndSplitAxis1(float s, const vec3& bMin, const vec3& bMax, 
                                            vec3& nMaxL, vec3& nMinR, u32& axis)
{
    nMaxL = vec3(bMax.x, s, bMax.z);
    nMinR = vec3(bMin.x, s, bMin.z);
    axis  = 2;
}

FDEVICE INLINE void getNewAABBAndSplitAxis2(float s, const vec3& bMin, const vec3& bMax, 
                                            vec3& nMaxL, vec3& nMinR, u32& axis)
{
    nMaxL = vec3(bMax.x, bMax.y, s);
    nMinR = vec3(bMin.x, bMin.y, s);
    axis  = 0;
}

using fGetNewAABBAxis = void (*)(float, const vec3&, const vec3&, vec3&, vec3&, u32&);
CONSTANT fGetNewAABBAxis c_getNewAABBAxis[3] =
{
    &getNewAABBAndSplitAxis0,
    &getNewAABBAndSplitAxis1,
    &getNewAABBAndSplitAxis2
};

// NOTE: Deliberately pass bMin and bMax by value as stack memory is used by stLeft and stRight.
DEVICE void splitOrAdd( bmTreeNode* node, vec3 bMin, vec3 bMax, u32 splitAxis, u32 ndepth,
                        bmStackNode* stLeft, bmStackNode* stRight, bmStore<bmTreeNode>* nodeStore )
{
    assert( node );
    assert( nodeStore );
    node->split( nodeStore ); // this creates 2 new tree nodes for node
    assert( node->m_left );
    assert( node->m_right );
    float s = .5f*(bMax[splitAxis]+bMin[splitAxis]);
    vec3 lMax, rMin;
    u32 naxis;
    c_getNewAABBAxis[splitAxis]( s, bMin, bMax, lMax, rMin, naxis );
    stLeft->init ( node->m_left,  bMin, lMax, naxis, ndepth );
    stRight->init( node->m_right, rMin, bMax, naxis, ndepth );
    //switch ( splitAxis )
    //{
    //case 0:
    //    stLeft->init ( node->m_left,  bMin, vec3(s, bMax.y, bMax.z), 1, ndepth );
    //    stRight->init( node->m_right, vec3(s, bMin.y, bMin.z), bMax, 1, ndepth );
    //    return;
    //
    //case 1:
    //    stLeft->init ( node->m_left,  bMin, vec3(bMax.x, s, bMax.z), 2, ndepth );
    //    stRight->init( node->m_right, vec3(bMin.x, s, bMin.z), bMax, 2, ndepth );
    //    return;

    //case 2:
    //    stLeft->init ( node->m_left,  bMin, vec3(bMax.x, bMax.y, s), 0, ndepth );
    //    stRight->init( node->m_right, vec3(bMin.x, bMin.y, s), bMax, 0, ndepth );
    //    return;
    //}
    //assert(false);
}

GLOBAL void bmInsertTriangleInTree( const vec3* vertices, const uint3* faces, 
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
    u32 tx   = tIdx.x;
    u32 fIdx = bIdx.x * bDim.x + tx;
    if ( fIdx >= numFaces ) return;
    uint3 id  = faces[fIdx];
    vec3 v[3] = { vertices[id.x], vertices[id.y], vertices[id.z] };

    // obtain aabb of triangle
    vec3 triMin, triMax;
    for ( int i=0; i<3; ++i )
    {
        triMin[i] = _min(v[0][i], _min(v[1][i], v[2][i]));
        triMax[i] = _max(v[0][i], _max(v[1][i], v[2][i]));
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
    #if _DEBUG
        bmValidateAABB( triMin, triMax );
    #endif
        if ( stNode->intersect( triMin, triMax ) )
        {
           if ( stNode->m_depth == BUILD_TREE_MAX_DEPTH-1 )
           {
           #if _DEBUG
               bmValidateAABB( stNode->m_min, stNode->m_max );
           //    bmPrintAABB( stNode->m_min, stNode->m_max );
           #endif
                stNode->m_node->insertFace( faceStore, faceGroupStore, meshIdx, id, mat );
            }
            else
            {
                assert( top+2 < BUILD_TREE_MAX_DEPTH );
                bmStackNode* l = &stack[++top];
                bmStackNode* r = &stack[++top];
                splitOrAdd( stNode->m_node, stNode->m_min, stNode->m_max, stNode->m_splitAxis, stNode->m_depth+1, l, r, nodeStore );
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

    bmTreeNode* t_rootNode             = (bmTreeNode*) treeRootNode;
    bmStore<bmFace>* t_faceStore       = (bmStore<bmFace>*) faceStore;
    bmStore<bmFace*>* t_faceGroupStore = (bmStore<bmFace*>*) faceGroupStore;
    bmStore<bmTreeNode>* t_nodeStore   = (bmStore<bmTreeNode>*) nodeStore;
    bmMaterial* t_material             = (bmMaterial*)material;

#if CUDA
    dim3 blocks ( (numFaces+BUILD_TREE_THREADS-1)/BUILD_TREE_THREADS );
    dim3 threads( BUILD_TREE_THREADS );
    bmInsertTriangleInTree<<< blocks, threads >>>
    ( 
        vertices, (uint3*)indices, numFaces, meshIdx,
        bMin, bMax, 
        t_rootNode, t_faceStore, t_faceGroupStore, 
        t_nodeStore, t_material
    );
#else
    bDim.x  = BUILD_TREE_THREADS;
    u32 blocks  = (numFaces+BUILD_TREE_THREADS-1)/BUILD_TREE_THREADS;
    u32 threads = BUILD_TREE_THREADS;
    for ( u32 b=0; b<blocks; b++ )
    {
        bIdx.x = b;
        for ( u32 t=0; t<threads; t++ )
        {
            tIdx.x = t;
            bmInsertTriangleInTree
            ( 
                vertices, (uint3*)indices, numFaces, meshIdx,
                bMin, bMax, 
                t_rootNode, t_faceStore, t_faceGroupStore, 
                t_nodeStore, t_material
            );
        }
    }
#endif

}

// -------- bmMarchKernel -----------------------------------------------------------------------------------------------------------


GLOBAL void bmMarchKernel(bmTreeNode* root, vec3 bMin, vec3 bMax,
                              const vec3* initialRays, u32 numRays,
                              u32* buffer, vec3 eye, mat3 orient,
                              const StaticMeshData* meshDataPtrs, u32 numMeshes )
{
    assert(root && initialRays && buffer && meshDataPtrs);

    u32 i = bIdx.x * bDim.x + tIdx.x;
    i = _min(i, numRays-1);

    vec3 dir = initialRays[i];
    dir = orient*dir;
    vec3 invDir(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);

    bmStackNode stack[BUILD_TREE_MAX_DEPTH];
    bmStackNode* st = &stack[0];
    st->init( root, bMin, bMax, 0, 0 );
    i32 top  = 0;
    float dClosest = FLT_MAX;
    bmFace* fClosest;
    float tU;
    float tV;

    do
    {
        assert( top >= 0 );
        st = &stack[top--];
        bmTreeNode* curNode = st->m_node;
        assert(st);
        assert(curNode);

        //if ( !curNode->m_right && !curNode->m_faces )
        //    continue;

        // check ray box intersect
        float boxDist = bmBoxRayIntersect(st->m_min, st->m_max, eye, invDir);
        if ( boxDist != FLT_MAX )
        {
            if ( !curNode->m_right ) // is leaf
            {
                if ( curNode->m_faces )
                {
                    // check intersections ray triangle list
                    u32 maxLoop = _min((u32)MAX_FACES_PER_BOX, curNode->m_faceInsertIdx);
                    for ( u32 i=0; i<maxLoop; i++ )
                    {
                        bmFace* face = curNode->m_faces[i];
                        float u, v;
                        float d = face->intersect(eye, dir, meshDataPtrs, numMeshes, u, v);
                        if ( d < dClosest )
                        {
                            dClosest = d;
                            fClosest = face;
                            tU = u;
                            tV = v;
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
                vec3 _min  = st->m_min;
                vec3 _max  = st->m_max;
                float s    = .5f*(_max[axis]+_min[axis]);
                vec3 p     = eye + boxDist*dir;
                if ( p[axis] < s ) // on 'left' side
                {
                    // push right first
                    float t = _min[axis];    // remember old _min[axis]
                    _min[axis] = s;          // overwrite
                    st = &stack[++top];      // push on stack
                    st->init(curNode->m_right, _min, _max, naxis, 0);
                    // push left now
                    _min[axis] = t;         // restore _min[axis]
                    _max[axis] = s;         // set _max[axis] for left
                    st = &stack[++top];     // push
                    st->init(curNode->m_left, _min, _max, naxis, 0);
                }
                else // on 'right' side
                {
                    // push left first
                    float t = _max[axis];    // remember old _min[axis]
                    _max[axis] = s;          // set _max[axis] for left
                    st = &stack[++top];      // push on stack
                    st->init(curNode->m_left, _min, _max, naxis, 0);
                    // push right now
                    _min[axis] = s;          // set _min[axis] split value       
                    _max[axis] = t;          // restore _max[axis] value
                    st = &stack[++top];      // push
                    st->init(curNode->m_right, _min, _max, naxis, 0);
                }
            }
        }
    } while ( top >= 0 );
    
    if ( dClosest != FLT_MAX )
    {
        assert( fClosest );
        vec4 n = fClosest->interpolate( tU, tV, meshDataPtrs, 1 );
        n.w = 0.f;
        n = normalize( n );
        buffer[i] = (u32)(fabs(n.z)*255) << 16;
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
#if CUDA
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
#else
    bDim.x = MARCH_THREADS;
    u32 blocks  = ( (numRays+MARCH_THREADS-1)/MARCH_THREADS );
    u32 threads = ( MARCH_THREADS );
    for ( u32 b=0; b<blocks; b++ )
    {
        bIdx.x = b;
        for ( u32 t=0; t<threads; t++ )
        {
            tIdx.x = t;
            bmMarchKernel
            ( 
                (bmTreeNode*)root,
                bMin, bMax,
                initialRays, numRays,
                buffer, eye, orient,
                meshData, numMeshes
            );
        }
    }
    
#endif
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