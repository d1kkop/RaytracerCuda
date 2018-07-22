#include "BuildTree.cuh"
#include "CudaHelp.h"

#define NUM_RESET_THREADS 256
#define MAX_HASH_ELEMENTS 65536
#define NUM_FACES_PER_CELL 32
#define MAX_SEARCH_ITERS 5
#define CELL_RES 1


// -------- Support ---------------------------------------------------------------------------------------------------------------

FDEVICE u32 bmHash(float f)
{
    u32 i = (u32) (f / CELL_RES);
    u64 rt = 1ULL<<63;
  //  #pragma unroll
    for ( u32 i=0; i<10; ++i )
    {
        i %= rt;
        rt -= 1992317;
    }
    i %= MAX_HASH_ELEMENTS;
    return (u32)abs(i);
}

FDEVICE u32 bmHash3(vec3 v)
{
    u32 h = bmHash(v.x) + bmHash(v.y) + bmHash(v.z);
    return h;
}

// -------- bmCell ---------------------------------------------------------------------------------------------------------------


struct bmCell
{
    bmFace m_faces[NUM_FACES_PER_CELL];
    vec3 m_min, m_max;
    u32 m_faceIdx;

    FDEVICE void reset()
    {
        m_faceIdx = 0;
    }

    FDEVICE void addFace(uint3 idx, u32 meshIdx, bmMaterial* mat)
    {
        u32 fIdx = atomicAdd2<u32>( &m_faceIdx, 1 );
        if ( fIdx < NUM_FACES_PER_CELL )
        {
            bmFace* f     = &m_faces[fIdx];
            f->m_index    = make_uint4(idx.x, idx.y, idx.z, meshIdx);
            f->m_material = mat;
        }
    }
};



// -------- bmResetSpaceKernel -----------------------------------------------------------------------------------------------------------

GLOBAL void bmResetSpaceKernel( bmCell* cells, u32 numCells )
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    i = _min(i, numCells-1);
    cells[i].reset();
}


extern "C"
void bmResetSpace( void* cells, u32 numCells )
{
    assert( cells );

#if CUDA
    bmCell* t_cells = (bmCell*) cells;
    dim3 blocks ( (numCells + NUM_RESET_THREADS-1)/NUM_RESET_THREADS );
    dim3 threads( NUM_RESET_THREADS );
    bmResetSpaceKernel<<< blocks, threads >>>
    ( 
        t_cells, numCells
    );
#endif
}


// -------- bmInsertTriangleInSpace -----------------------------------------------------------------------------------------------------------

GLOBAL void bmInsertTriangleInSpace( const vec3* vertices, 
                                         const uint3* faces, u32 numFaces, 
                                         bmMaterial* mat, u32 meshIdx, 
                                         bmCell* cells, u32 numCells )
{
    assert( vertices && faces );

    // face index
    u32 tx = threadIdx.x;
    u32 fIdx = blockIdx.x * blockDim.x + tx;
    if ( fIdx >= numFaces ) return;
    uint3 idx  = faces[fIdx];
    vec3 v[3]  = { vertices[idx.x], vertices[idx.y], vertices[idx.z] };

    // obtain aabb of triangle QQQ (could be precalculated)
    vec3 tmin, tmax;
//    #pragma unroll
    for ( int i=0; i<3; ++i )
    {
        tmin[i] = _min(v[0][i], _min(v[1][i], v[2][i]));
        tmax[i] = _max(v[0][i], _max(v[1][i], v[2][i]));
    }

    vec3 p[8] =
    {
        tmin,
        vec3(tmin.x,tmin.y,tmax.z),
        vec3(tmin.x,tmax.y,tmax.z),
        vec3(tmin.x,tmax.y,tmin.z),
        vec3(tmax.x,tmin.y,tmin.z),
        vec3(tmax.x,tmax.y,tmin.z),
        vec3(tmax.x,tmin.y,tmax.z),
        tmax
    };

    #pragma unroll
    for (const auto& vt : p)
    {
        i32 h = bmHash3( vt );
        cells[h].addFace( idx, meshIdx, mat );
    }

}


extern "C"
void bmInsertMeshInSpace( const vec3* vertices, 
                          const u32* indices, u32 numIndices, 
                          void* material, u32 meshIdx,
                          void* cells, u32 numCells )
{
  #if CUDA
     u32 numFaces = numIndices / 3;
    assert( numIndices%3 == 0 );

    dim3 blocks ( (numFaces+BUILD_TREE_THREADS-1)/BUILD_TREE_THREADS );
    dim3 threads( BUILD_TREE_THREADS );

    bmCell* t_cells   = (bmCell*)cells;
    uint3* t_faces    = (uint3*)indices;
    bmMaterial* t_mat = (bmMaterial*)material;

    bmInsertTriangleInSpace<<< blocks, threads >>>
        (
            vertices,
            t_faces, numFaces,
            t_mat, meshIdx,
            t_cells, numCells
            );
#endif // CUDA
    }

// -------- bmMarchKernelSpace -----------------------------------------------------------------------------------------------------------


GLOBAL void bmMarchKernelSpace( const vec3* initialRays, u32 numRays,
                                    u32* buffer, vec3 eye, mat3 orient,
                                    const StaticMeshData* meshDataPtrs, u32 numMeshes,
                                    bmCell* cells, u32 numCells )
{
    assert(initialRays && buffer && meshDataPtrs);

    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    i = _min(i, numRays-1);

    vec3 dir = initialRays[i];
    dir = orient*dir;
    vec3 invDir(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);

    float tU, tV;
    float dClosest = FLT_MAX;
    vec3 p = eye;
    
    u32 maxSearchIters = MAX_SEARCH_ITERS;
    do
    {
        u32 h = bmHash3( p );
        bmCell* c = cells + h;

        if ( p.x >= c->m_min.x && p.y >= c->m_min.y && p.z >= c->m_min.z &&
             p.x <= c->m_max.x && p.y <= c->m_max.y && p.z <= c->m_max.z )
        {
//            bmFace* faces = c->m_faces;

 //           #pragma unroll
            for ( u32 i=0; i<NUM_FACES_PER_CELL; i++ )
            {
                float u=0,v=0;
                float kDist  = 0; // faces[i].intersect(eye, dir, meshDataPtrs, numMeshes, u, v);
                bool bCloser = kDist < dClosest;
                dClosest = (bCloser ? kDist : dClosest);
                tU = (bCloser ? u : tU);
                tV = (bCloser ? v : tV);
            }

            if ( dClosest != FLT_MAX ) break;
        }

        float boxDist = bmBoxRayIntersect( c->m_min, c->m_max, p, invDir );
        p += dir*(boxDist + 0.001f);


    } while ( --maxSearchIters );


        // check ray box intersect
  //      float boxDist = bmBoxRayIntersect(st->m_min, st->m_max, eye, invDir);
   
    
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
void bmMarchSpace( const vec3* initialRays,
                   u32* buffer, u32 pitch, u32 width, u32 height,
                   const vec3& eye, const mat3& orient,
                   const StaticMeshData* meshData, u32 numMeshes,
                   void* cells, u32 numCells )
{
#if CUDA
    u32 numRays = width*height;
    bmCell* t_cells = (bmCell*)cells;
    dim3 blocks ( (numRays+MARCH_THREADS-1)/MARCH_THREADS );
    dim3 threads( MARCH_THREADS );
    bmMarchKernelSpace<<< blocks, threads >>>
    ( 
        initialRays, numRays,
        buffer, eye, orient,
        meshData, numMeshes,
        t_cells, numCells
    );
#endif
}


extern "C"
{
    u32 bmGetCellSize() { return sizeof(bmCell); }
}