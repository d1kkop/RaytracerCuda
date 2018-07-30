#include "CudaComon.cuh"

#define MAX_HITS_PER_RAY 16
#define NUM_FACES_IN_LEAF 64

#define IS_LEAF( idx ) ((idx>>31)==1)
#define INVALID_INDEX ((u32)-1)
#define VALID( idx ) (idx!=INVALID_INDEX)
#define NUM_TRIANGLES(idx) (idx&0x7FFFFFF)


// -------- bmMarchKernel -----------------------------------------------------------------------------------------------------------


struct bmFaceCluster
{
    bmFace* faces[NUM_FACES_IN_LEAF];
};


struct bmBvhNode
{
    vec3 hs, cp;
    u32 left, right;

    FDEVICE bool isLeaf() const
    {
        return IS_LEAF( left );
    }

    FDEVICE u32 numFaces() const
    {
        assert( isLeaf() );
        return NUM_TRIANGLES( left );
    }

    FDEVICE bmFace* getFace(bmFaceCluster* faceClusters, u32 idx) const
    {
        assert( isLeaf() );
        bmFaceCluster* fc = faceClusters + right;
        return fc->faces[ idx ];
    }
};

struct bmRay
{
    vec3 o, d;
};

struct bmRayBox
{
    bmRay* ray;
    bmBvhNode* node;
};

struct bmRayFace
{
    bmRay* ray;
    bmFace* face;
};

struct bmRayFaceHitCluster
{
    bmRayFaceHitResult results[MAX_HITS_PER_RAY];
    u32 count;
};


GLOBAL void bmRayBoxKernel( bmStore<bmRayBox>* rayBoxQueue, 
                            bmStore<u32>* leafQueue, 
                            bmBvhNode* bvhNodes )
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= rayBoxQueue->m_max ) return;
	bmRayBox* rb = rayBoxQueue->get(i);
    vec3 o   = rb->ray->o;
    vec3 d   = rb->ray->d;
    vec3 cp  = rb->node->cp;
    vec3 hs  = rb->node->hs;
    vec3 invDir(1.f/d.x, 1.f/d.y, 1.f/d.z);
    if ( bmBoxRayIntersect(cp-hs, cp+hs, o, invDir) )
    {
        if ( rb->node->isLeaf() )
        {
            u32* leafTestIdx = leafQueue->getNew(1);
            *leafTestIdx = i;
        }
        else
        {
            // both must be valid
            assert( VALID(rb->node->left) && VALID(rb->node->right) );
            // replace existing with left
            rb->node = bvhNodes + rb->node->left;
            // get new ray/box memory and store right
            bmRayBox* rbNew = rayBoxQueue->getNew(1);
            rbNew->ray  = rb->ray;
            rbNew->node = bvhNodes + rb->node->right;
        }
    }
}

GLOBAL void bmLeafExpandKernel( bmStore<bmRayBox>* rayBoxQueue, 
                                bmStore<u32>* leafQueue, 
                                bmStore<bmRayFace>* rayFaceQueue,
                                bmBvhNode* bvhNodes,
                                bmFaceCluster* faceClusters )
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= leafQueue->m_max ) return;
	u32 rayBoxIdx = *leafQueue->get(i);
    bmRayBox* rb = rayBoxQueue->get(rayBoxIdx);
    // It is already known that the ray intersects the leaf-box, so queue all ray/triangle tests.
    assert( rb->isLeaf() );
    u32 numFaces  = rb->node->numFaces();
    bmRayFace* rf = rayFaceQueue->getNew( numFaces );
    for ( u32 k=0; k<numFaces; ++k )
    {
        rf->ray  = rb->ray;
        rf->face = rb->node->getFace( faceClusters, k );
    }
}


GLOBAL void bmFaceTestKernel( bmRay* rays,
                              bmStore<bmRayFace>* rayFaceQueue,
                              bmFaceCluster* faceClusters, 
                              bmRayFaceHitCluster* hitResultClusters,
                              StaticMeshData* meshDataPtrs )

{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= rayFaceQueue->m_max ) return;
	bmRayFace* rf = rayFaceQueue->get(i);
    vec3 o  = rf->ray->o;
    vec3 d  = rf->ray->d;
    float u, v;
    float dist = bmFaceRayIntersect( rf->face, o, d, meshDataPtrs, u, v );
    if ( dist != FLT_MAX )
    {
        bmRayFaceHitCluster* hitCluster = hitResultClusters + (rf->ray - rays);
        u32 curHitIdx = atomicAdd2<u32>( &hitCluster->count, 1 );
        if ( curHitIdx < MAX_HITS_PER_RAY )
        {
            bmRayFaceHitResult* result = hitCluster->results + curHitIdx;
            result->face = rf->face;
            result->dist = dist;
            result->u = u;
            result->v = v;
        }
    }
}


GLOBAL void bmFindClosestHit( bmRayFaceHitCluster* hitResultClusters, u32 numRays,
                              StaticMeshData* meshDataPtrs,
                              u32* buffer,
                              bmHitCallback cb )

{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= numRays ) return;
	bmRayFaceHitCluster* hitCluster = hitResultClusters + i;
    u32 count = _min( hitCluster->count, (u32)MAX_HITS_PER_RAY );
    float fDist = FLT_MAX;
    bmRayFaceHitResult* rf = nullptr;
    for ( u32 k=0; k<count; ++k )
    {
        bmRayFaceHitResult* result = hitCluster->results + k;
        if ( result->dist < fDist )
        {
            fDist = result->dist;
            rf = result;
        }
    }
    cb( i, rf, meshDataPtrs, buffer );
}


DEVICE void bmShadeNormal(u32 i, const bmRayFaceHitResult* res, const StaticMeshData* meshDataPtrs, u32* buffer)
{
    vec4 n = bmFaceInterpolate( res->face, res->u, res->v, meshDataPtrs, VERTEX_DATA_NORMAL );
    n = normalize(n);
    buffer[i] = (u32)(fabs(n.z)*255) << 16;

}


extern "C"
void bmMarchProgressive(void* rays, u32 numRays, 
                        void* rayBoxQueue,
                        void* leafQueue,
                        void* rayFaceQueue,
                        void* bvhNodes,
                        void* faceClusters,
                        void* hitResultClusters,
                        void* meshDataPtrs,
                        bmHitCallback cb)
{
   


#if CUDA

    bmFindClosestHit<<< 1, 1 >>> (
        nullptr, 
        0, 
        nullptr,
        nullptr,
        &bmShadeNormal
        );

#else
   
#endif
}


extern "C"
{

}