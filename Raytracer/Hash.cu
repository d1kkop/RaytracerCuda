#include "BuildTree.cuh"

#define NUM_RESET_THREADS 256
#define BUILD_CELLS_THREADS BUILD_TREE_THREADS
#define MARCH_CELLS_THREADS MARCH_THREADS
#define NUM_FACES_PER_CELL 32
#define CELL_RES 0.05f
#define INV_CELL_RES (1.f/CELL_RES)
#define CELL_PINCH_TROUGH_EPSILON (CELL_RES*0.001f)
#define MAX_SEARCH_ITERS 100

// -------- Support ---------------------------------------------------------------------------------------------------------------

// https://en.wikipedia.org/wiki/List_of_hash_functions
// Fletcher_16
FDEVICE u32 bmHash(u32 h)
{
    u16 sum1 = 0;
    u16 sum2 = 0;
    int index;
    u8* data = (u8*)&h;
    #pragma unroll
    for ( index = 0; index < 4; ++index )
    {
        sum1 = (sum1 + data[index]) % 255;
        sum2 = (sum2 + sum1) % 255;
    }
    h = (sum2 << 8) | sum1;
    assert( h < MAX_HASH_ELEMENTS );
    return h;
}

FDEVICE u32 bmHashF(float f)
{
    u32 i = (u32)(f * INV_CELL_RES);
    return bmHash( i );
}

FDEVICE u32 bmHash3(uint3 v)
{
    u32 h = bmHash(v.x) + bmHash(v.y) + bmHash(v.z);
    h = h % MAX_HASH_ELEMENTS;
 //   printf("h %d\n", h);
    return h;
}

FDEVICE u32 bmHash3(const vec3& v)
{
    u32 h = bmHashF(v.x) + bmHashF(v.y) + bmHashF(v.z);
    h = h % MAX_HASH_ELEMENTS;
 //   printf("h %d\n", h);
    return h;
}

FDEVICE i32 bmMap(float f)
{
    return (i32) floorf(f*INV_CELL_RES);
}

FDEVICE int3 bmMap3(const vec3& v)
{
    return make_int3( bmMap(v.x), bmMap(v.y), bmMap(v.z) );
}

// -------- bmCell ---------------------------------------------------------------------------------------------------------------


struct bmCell
{
    bmFace m_faces[NUM_FACES_PER_CELL];
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

GLOBAL void bmResetSpaceKernel( bmCell* cells )
{
    u32 i = bIdx.x * bDim.x + tIdx.x;
    i = _min(i, (u32)MAX_HASH_ELEMENTS-1);
    cells[i].reset();
}


extern "C"
void bmResetSpace( void* cells )
{
    assert( cells );
    bmCell* t_cells = (bmCell*) cells;
    dim3 blocks ( (MAX_HASH_ELEMENTS + NUM_RESET_THREADS-1)/NUM_RESET_THREADS );
    dim3 threads( NUM_RESET_THREADS );
#if CUDA
    bmResetSpaceKernel<<< blocks, threads >>>
    ( 
        t_cells
    );
#else
    bDim.x = NUM_RESET_THREADS;
    for ( u32 b=0; b<blocks.x; b++ )
    {
        bIdx.x = b;
        for ( u32 t=0; t<threads.x; t++ )
        {
            tIdx.x = t;
            bmResetSpaceKernel( t_cells );
        }
    }
#endif
}


// -------- bmInsertTriangleInSpace -----------------------------------------------------------------------------------------------------------

GLOBAL void bmInsertTriangleInSpace( const vec3* vertices, 
                                     const uint3* faces, u32 numFaces, 
                                     bmMaterial* mat, u32 meshIdx, 
                                     bmCell* cells )
{
    assert( vertices && faces );

    // face index
    u32 tx   = tIdx.x;
    u32 fIdx = bIdx.x * bDim.x + tx;
    if ( fIdx >= numFaces ) return;
    uint3 idx  = faces[fIdx];
    vec3 v[3]  = { vertices[idx.x], vertices[idx.y], vertices[idx.z] };

    // obtain aabb of triangle QQQ (could be precalculated)
    vec3 tmin, tmax;
    #pragma unroll
    for ( int i=0; i<3; ++i )
    {
        tmin[i] = _min(v[0][i], _min(v[1][i], v[2][i]));
        tmax[i] = _max(v[0][i], _max(v[1][i], v[2][i]));
    }

    i32 x1 = bmMap(tmin.x);
    i32 y1 = bmMap(tmin.y);
    i32 z1 = bmMap(tmin.z);
    i32 x2 = bmMap(tmax.x);
    i32 y2 = bmMap(tmax.y);
    i32 z2 = bmMap(tmax.z);

    for ( ; z1 <= z2; ++z1 )
    for ( ; y1 <= y2; ++y1 )
    for ( ; x1 <= x2; ++x1 )
    {
        uint3 p = make_uint3(x1, y1, z1);
        u32 h   = bmHash3( p );
        cells[h].addFace( idx, meshIdx, mat );
    }
}


extern "C"
void bmInsertMeshInSpace( const vec3* vertices, 
                          const u32* indices, u32 numIndices, 
                          void* material, u32 meshIdx,
                          void* cells )
{
    u32 numFaces = numIndices / 3;
    assert( numIndices%3 == 0 );

    dim3 blocks ( (numFaces+BUILD_CELLS_THREADS-1)/BUILD_CELLS_THREADS );
    dim3 threads( BUILD_CELLS_THREADS );

    bmCell* t_cells   = (bmCell*)cells;
    uint3* t_faces    = (uint3*)indices;
    bmMaterial* t_mat = (bmMaterial*)material;

  #if CUDA
    bmInsertTriangleInSpace<<< blocks, threads >>>
    (
        vertices,
        t_faces, numFaces,
        t_mat, meshIdx,
        t_cells
    );
#else
    bDim.x = BUILD_CELLS_THREADS;
    for ( u32 b=0; b<blocks.x; b++ )
    {
        bIdx.x = b;
        for ( u32 t=0; t<threads.x; t++ )
        {
            tIdx.x = t;
            bmInsertTriangleInSpace
            (
                vertices,
                t_faces, numFaces,
                t_mat, meshIdx,
                t_cells
            );
        }
    }

    printf("Begin\n");
    for ( int i = 0; i < MAX_HASH_ELEMENTS; ++i )
    {
        if ( t_cells[i].m_faceIdx != 0 )
            printf("Cell %d, faceIdx %d\n", i, t_cells[i].m_faceIdx);
    }
#endif
    }

// -------- bmMarchKernelSpace -----------------------------------------------------------------------------------------------------------


GLOBAL void bmMarchKernelSpace( const vec3* initialRays, u32 numRays,
                                    u32* buffer, vec3 eye, mat3 orient,
                                    const StaticMeshData* meshDataPtrs,
                                    bmCell* cells )
{
    assert(initialRays && buffer && meshDataPtrs);

    u32 i = bIdx.x * bDim.x + tIdx.x;
    i = _min(i, numRays-1);

    vec3 dir = initialRays[i];
    dir = orient*dir;
    vec3 invDir(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);

    float tU=0, tV=0;
    float dClosest = FLT_MAX;
    bmFace* fClosest=nullptr;
    vec3 p = eye;
    
    u32 maxSearchIters = MAX_SEARCH_ITERS;
    #pragma unroll
    do
    {
        int3 cp = bmMap3 ( p );
        u32 h   = bmHash3( make_uint3(cp.x,cp.y,cp.z) );
        bmCell* c = cells + h;

        // Check if p is actually inside of cell or that the cell arose from hash collision.
        if (c->m_faceIdx>0 &&
            (p.x >= cp.x) && 
            (p.x <= cp.x+CELL_RES) && 
            (p.y >= cp.y) && 
            (p.y <= cp.y+CELL_RES) && 
            (p.z >= cp.z) &&
            (p.z <= cp.z+CELL_RES))
        {
            bmFace* faces = c->m_faces;

            u32 maxLoop = _min( (u32)NUM_FACES_PER_CELL, c->m_faceIdx );
            #pragma unroll
            for ( u32 i=0; i<NUM_FACES_PER_CELL; i++ )
            {
                float u=0,v=0;
                float kDist  = bmFaceRayIntersect(faces + i, eye, dir, meshDataPtrs, u, v);
                bool bCloser = kDist < dClosest;
                dClosest = (bCloser ? kDist : dClosest);
                tU = (bCloser ? u : tU);
                tV = (bCloser ? v : tV);
                fClosest = (bCloser ? &faces[i] : fClosest);
            }

            if ( dClosest != FLT_MAX ) break;
        }

        vec3 bMin = vec3( cp.x, cp.y, cp.z )*CELL_RES;
        vec3 bMax = bMin + vec3(CELL_RES, CELL_RES, CELL_RES);
        float boxDist = bmBoxRayIntersectNoZero( bMin, bMax, p, invDir );
        p += dir*(boxDist + CELL_PINCH_TROUGH_EPSILON);


    } while ( --maxSearchIters );

    if ( dClosest != FLT_MAX )
    {
        assert( fClosest );
        vec4 n = bmFaceInterpolate( fClosest, tU, tV, meshDataPtrs, 1 );
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
void bmMarchSpace( const vec3* initialRays,
                   u32* buffer, u32 pitch, u32 width, u32 height,
                   const vec3& eye, const mat3& orient,
                   const StaticMeshData* meshData,
                   void* cells )
{
    u32 numRays = width*height;
    bmCell* t_cells = (bmCell*)cells;
    dim3 blocks ( (numRays+MARCH_CELLS_THREADS-1)/MARCH_CELLS_THREADS );
    dim3 threads( MARCH_CELLS_THREADS );
#if CUDA
    bmMarchKernelSpace<<< blocks, threads >>>
    ( 
        initialRays, numRays,
        buffer, eye, orient,
        meshData,
        t_cells
    );
#else
    bDim.x = MARCH_CELLS_THREADS;
    for ( u32 b=0; b<blocks.x; b++ )
    {
        bIdx.x = b;
        for ( u32 t=0; t<threads.x; t++ )
        {
            tIdx.x = t;
            bmMarchKernelSpace
            ( 
                initialRays, numRays,
                buffer, eye, orient,
                meshData,
                t_cells
            );
        }
    }
#endif
}


extern "C"
{
    u32 bmGetCellSize() { return sizeof(bmCell); }
}