#include "BuildTree.cuh"
#include "BoxTriangle.cuh"

// -------- bmTreeNode -----------------------------------------------------------------------------------------------------------


FDEVICE void bmTreeNode::split(bmStore<bmTreeNode>* store)
{
    u32 bEval = m_left?1:0;
    bEval += m_right?1:0;
    if ( bEval != 2 )
    {
        // NOTE: If left and/or right are not assigned due to atomic compare & swap memory is LEAKED.
        // However, the leaked memory is reclaimed every frame.
        bmTreeNode* leftAndRight = store->getNew(2);
        atomicCAS2<u64>( (u64*)&m_left, (u64)0, (u64)leftAndRight );
        atomicCAS2<u64>( (u64*)&m_right, (u64)0, (u64)(leftAndRight+1) );
    }
}

FDEVICE void bmTreeNode::split2(bool b1, bool b2, bmStore<bmTreeNode>* store)
{
    if ( b1 && !m_left )
    {
        bmTreeNode* left = store->getNew(1);
        atomicCAS2<u64>( (u64*)&m_left, (u64)0, (u64)left );
    }
    if ( b2 && !m_right )
    {
        bmTreeNode* right = store->getNew(1);
        atomicCAS2<u64>( (u64*)&m_right, (u64)0, (u64)right );
    }
    THREAD_FENCE();
}

FDEVICE void bmTreeNode::insertFace( bmStore<bmFace>* faceStore, bmStore<bmFace*>* faceGroupStore, u32 meshIdx, uint3 faceIdx, bmMaterial* mat )
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
    else printf("Num faces %d\n", storeFidx);
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
    return bmAABBOverlap( triMin, triMax, m_min, m_max );
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


GLOBAL void bmInsertTriangleInTree( const vec3* RESTRICT vertices, const uint3* RESTRICT faces, 
                                    u32 numFaces, u32 meshIdx,
                                    vec3 bMin, vec3 bMax, 
                                    bmTreeNode* RESTRICT root, 
                                    bmStore<bmFace>* RESTRICT faceStore,
                                    bmStore<bmFace*>* RESTRICT faceGroupStore,
                                    bmStore<bmTreeNode>* RESTRICT nodeStore, 
                                    bmMaterial* RESTRICT mat )
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
    //vec3 triMin, triMax;
    //for ( int i=0; i<3; ++i )
    //{
    //    triMin[i] = _min(v[0][i], _min(v[1][i], v[2][i]));
    //    triMax[i] = _max(v[0][i], _max(v[1][i], v[2][i]));
    //}
//#if _DEBUG
//    bmValidateAABB( triMin, triMax );
//#endif

 //   __shared__ bmStackNode stack_shared[BUILD_TREE_THREADS][TREE_SEARCH_DEPTH];
 //   bmStackNode* stack = stack_shared[tx];
    bmStackNode dummy;
    bmStackNode stack[TREE_SEARCH_DEPTH];
    stack[0].init( root, bMin, bMax, 0, 0 );
    i32 top = 0;

    do
    {
        bmStackNode* RESTRICT stNode = &stack[top--];
        assert( stNode && stNode->m_node );
        bmTreeNode* RESTRICT node = stNode->m_node;

        vec3 bSize = stNode->m_max - stNode->m_min;
        float dMin = _min(bSize.x, _min(bSize.y, bSize.z));

        if ( (dMin < MIN_LEAF_SIZE) || (stNode->m_depth == BUILD_TREE_MAX_DEPTH-1) ) // <-- correct, must be build_tree_max_depth
        {
        #if _DEBUG
            bmValidateAABB( stNode->m_min, stNode->m_max );
      //    bmPrintAABB( stNode->m_min, stNode->m_max );
        #endif
            stNode->m_node->insertFace( faceStore, faceGroupStore, meshIdx, id, mat );
        }
        else
        {
            assert( top+2 < TREE_SEARCH_DEPTH );
            vec3 stMin = stNode->m_min;
            vec3 stMax = stNode->m_max;
            u32 ndepth = stNode->m_depth+1;
            u32 spAxis = stNode->m_splitAxis;
            u32 nAxis  = (spAxis+1)%3;
            float s    = .5f*(stMin[spAxis]+stMax[spAxis]);
            vec3 lMax  = stMax;
            vec3 rMin  = stMin;
            lMax[spAxis] = s;
            rMin[spAxis] = s;
            vec3 bc = (lMax+stMin)*.5f;
            vec3 hs = (lMax-stMin)*.5f;
            bool b1 = triBoxOverlap( &bc.x, &hs.x, v );
            bc = (stMax+rMin)*.5f;
            hs = (stMax-rMin)*.5f;
            bool b2 = triBoxOverlap( &bc.x, &hs.x, v );
            // bool b1 = bmAABBOverlap2( triMin, triMax, stMin, lMax );
            // bool b2 = bmAABBOverlap2( triMin, triMax, rMin, stMax );
            node->split2( b1, b2, nodeStore );
            bmTreeNode* RESTRICT nodeL = node->m_left;
            bmTreeNode* RESTRICT nodeR = node->m_right;
            // push left
         //   stNode = b1?&stack[++top]:&dummy;
            if ( b1 )
            {
                stNode = &stack[++top];
                stNode->m_min   = stMin;
                stNode->m_max   = lMax;
                stNode->m_node  = nodeL;
                stNode->m_depth = ndepth;
                stNode->m_splitAxis = nAxis;
            }
            // push right
            //stNode = b2?&stack[++top]:&dummy;
            if ( b2 )
            {
                stNode = &stack[++top];
                stNode->m_min    = rMin;
                stNode->m_max    = stMax;
                stNode->m_node   = nodeR;
                stNode->m_depth  = ndepth;
                stNode->m_splitAxis = nAxis;
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
    // Debug info
    {
        bmStackNode dummy;
        bmStackNode stack[TREE_SEARCH_DEPTH];
        stack[0].init(t_rootNode, bMin, bMax, 0, 0);
        i32 top = 0;
        u64 avgDepth = 0;
        u64 numLeafs = 0;
        u32 maxDepth = 0;
        u64 numFaces = 0;
        u64 avgFacesPerLeaf = 0;
        do
        {
            bmStackNode* RESTRICT stNode = &stack[top--];
            assert(stNode && stNode->m_node);
            bmTreeNode* RESTRICT node = stNode->m_node;
            if ( !node->m_right && !node->m_left ) // is leaf
            {
                avgDepth += stNode->m_depth;
                numLeafs += 1;
                maxDepth = _max(maxDepth, stNode->m_depth);
                numFaces += node->m_faceInsertIdx;
                avgFacesPerLeaf += node->m_faceInsertIdx;
                vec3 dt = stNode->m_max-stNode->m_min;
                printf("Leaf-> Depth %d Size: %.3f %.3f %.3f\n", stNode->m_depth, dt.x, dt.y, dt.z);
            }
            else
            {
                u32 spAxis = stNode->m_splitAxis;
                u32 nAxis  = (spAxis+1)%3;
                u32 depth  = stNode->m_depth+1;
                vec3 stMin = stNode->m_min;
                vec3 stMax = stNode->m_max;
                vec3 lmax  = stMax;
                vec3 rmin  = stMin;
                float s = (stMax[spAxis]+stMin[spAxis])*.5f;
                lmax[spAxis] = s;
                rmin[spAxis] = s;
                if ( node->m_left )
                {
                    ++top;
                    stack[top].init(node->m_left, stMin, lmax, nAxis, depth);
                }
                if ( node->m_right )
                {
                    ++top;
                    stack[top].init(node->m_right, rmin, stMax, nAxis, depth);
                }
            }
        }
        while ( top >= 0 );
        printf("Tree Faces: %d, Nodes %d, FacePtrs %d\n", t_faceStore->m_top, t_nodeStore->m_top, t_faceGroupStore->m_top);
        printf("MaxDepth: %d, AvgDepth %zd, NumLeafs %zd, numFaces %zd, avgFacesPerLeaf %zd\n", maxDepth, avgDepth/numLeafs, numLeafs, numFaces, avgFacesPerLeaf/numLeafs);
    }
#endif
}

// -------- bmMarchKernel -----------------------------------------------------------------------------------------------------------


GLOBAL void bmMarchKernel(bmTreeNode* RESTRICT root, vec3 bMin, vec3 bMax,
                              const vec3* initialRays, u32 numRays,
                              u32* buffer, vec3 eye, mat3 orient,
                              const StaticMeshData* RESTRICT meshDataPtrs )
{
    assert(root && initialRays && buffer && meshDataPtrs);

    u32 i = bIdx.x * bDim.x + tIdx.x;
    i = _min(i, numRays-1);

    vec3 dir = initialRays[i];
    dir = orient*dir;
    vec3 invDir(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);

//    __shared__ bmStackNode s_stack[BUILD_TREE_THREADS][TREE_SEARCH_DEPTH];
//    bmStackNode* stack = s_stack[tIdx.x];

    bmStackNode stack[TREE_SEARCH_DEPTH];
    bmStackNode* RESTRICT st = &stack[0];

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
        assert(st && curNode);

        //if ( !curNode->m_right && !curNode->m_faces )
        //    continue;

        // check ray box intersect
        float boxDist = bmBoxRayIntersect(st->m_min, st->m_max, eye, invDir);
        if ( boxDist != FLT_MAX )
        {
            if ( !curNode->m_right && !curNode->m_left ) // is leaf
            {
                if ( curNode->m_faces )
                {
                    // check intersections ray triangle list
                    u32 maxLoop = _min((u32)MAX_FACES_PER_BOX, curNode->m_faceInsertIdx);
                    for ( u32 i=0; i<maxLoop; i++ )
                    {
                        bmFace* face = curNode->m_faces[i];
                        float u, v;
                        float d = bmFaceRayIntersect( face, eye, dir, meshDataPtrs, u, v );
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
                assert( top+2 < TREE_SEARCH_DEPTH );
                vec3 stMin  = st->m_min;
                vec3 stMax  = st->m_max;
                u32 axis    = st->m_splitAxis;
                u32 naxis   = (axis+1)%3;
                float s     = .5f*(stMax[axis]+stMin[axis]);
                float p     = eye[axis] + boxDist*dir[axis];
                if ( p < s ) // on 'left' side
                {
                    // push right first
                    if ( curNode->m_right )
                    {
                        st = &stack[++top];
                        float t = stMin[axis];
                        stMin[axis] = s;
                        st->init(curNode->m_right, stMin, stMax, naxis, 0);
                        stMin[axis] = t; 
                    }
                    // push left now
                    if ( curNode->m_left )
                    {
                        st = &stack[++top];
                        stMax[axis] = s;
                        st->init(curNode->m_left, stMin, stMax, naxis, 0);
                    }
                }
                else // on 'right' side
                {
                    // push left first
                    if ( curNode->m_left )
                    {
                        st = &stack[++top];
                        float t = stMax[axis];
                        stMax[axis] = s;
                        st->init(curNode->m_left, stMin, stMax, naxis, 0);
                        stMax[axis] = t;
                    }
                    // push right now
                    if ( curNode->m_right )
                    {
                        st = &stack[++top]; 
                        stMin[axis] = s;   
                        st->init(curNode->m_right, stMin, stMax, naxis, 0);
                    }
                }
            }
        }
    } while ( top >= 0 );
    
    if ( dClosest != FLT_MAX )
    {
        assert( fClosest );
        vec4 n  = bmFaceInterpolate( fClosest, tV, tU, meshDataPtrs, 1 );
        vec3 nn = normalize( n );
        buffer[i] = (u32)((-nn.z)*255) << 16;
    }
    else
    {
        buffer[i] = 255<<8;
    }

 //   cudaDeviceSynchronize();
}


extern "C"
void bmMarch(const void* root, const vec3& bMin, const vec3& bMax, 
             const vec3* initialRays,
             u32* buffer, u32 pitch, u32 width, u32 height,
             const vec3& eye, const mat3& orient,
             const StaticMeshData* meshData)
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
        meshData
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
                meshData
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