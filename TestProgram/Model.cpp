#include "Model.h"
#include "../Raytracer/Beam.h"
#include "../Raytracer/Types.h"
#include "../Raytracer/Util.h"
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "glm/vec3.hpp"
#include "glm/mat3x3.hpp"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include <iostream>
#include <cassert>
using namespace Assimp;
using namespace std;
using namespace Beam;
using namespace glm;

namespace TestProgram
{
    float _min2(float a, float b) { return a < b ? a : b; }
    float _max2(float a, float b) { return a > b ? a : b; }
    vec3 _min2(const vec3& a, const vec3& b) { return vec3(_min2(a.x, b.x), _min2(a.y, b.y), _min2(a.z, b.z)); }
    vec3 _max2(const vec3& a, const vec3& b) { return vec3(_max2(a.x, b.x), _max2(a.y, b.y), _max2(a.z, b.z)); }

    bool Model::load(const std::string& name, sptr<IScene>& toScene, int numAdds)
    {
        // Create an instance of the Importer class
        Importer importer;
        // And have it read the given file with some example postprocessing
        // Usually - if speed is not the most important aspect for you - you'll
        // probably to request more postprocessing than we do in this example.
        const aiScene* scene = importer.ReadFile(name,
                                                 aiProcess_CalcTangentSpace       |
                                                 aiProcess_Triangulate            |
                                                 aiProcess_JoinIdenticalVertices  |
                                                 aiProcess_SortByPType);
        // If the import failed, report it
        if ( !scene ) {
            cout << importer.GetErrorString() << endl;
            assert(false);
            return false;
        }
        // Now we can access the file's contents.
        u32 totalVertices =0;
        u32 totalFaces =0;
        vec3 bMin(FLT_MAX);
        vec3 bMax(FLT_MIN);
        for (u32 i = 0; i < scene->mNumMeshes ; i++)
        {
        	const aiMesh* mesh  = scene->mMeshes[i];
            sptr<IMesh> gpuMesh = IMesh::create();
            for ( int j=0; j<numAdds; j++ )
                toScene->addMesh( gpuMesh );
            totalVertices += mesh->mNumVertices;
            totalFaces += mesh->mNumFaces;
            u32 err=0;
            // deform indices
            u32* indices = new u32[mesh->mNumFaces*3];
            for ( u32 i=0; i<mesh->mNumFaces; ++i )
            {
                assert(mesh->mFaces[i].mNumIndices==3);
                indices[i*3+0] = mesh->mFaces[i].mIndices[0];
                indices[i*3+1] = mesh->mFaces[i].mIndices[1];
                indices[i*3+2] = mesh->mFaces[i].mIndices[2];
                assert(indices[i*3+0]  < mesh->mNumVertices);
                assert(indices[i*3+1]  < mesh->mNumVertices);
                assert(indices[i*3+2]  < mesh->mNumVertices);
                vec3 v[3] =
                {
                    *(vec3*)&mesh->mVertices[indices[i*3+0]],
                    *(vec3*)&mesh->mVertices[indices[i*3+1]],
                    *(vec3*)&mesh->mVertices[indices[i*3+2]]
                };
                for ( auto& vs : v )
                {
                    bMin = _min2(bMin, vs);
                    bMax = _max2(bMax, vs);
                }
            }
            gpuMesh->setIndices(indices, mesh->mNumFaces*3 );
            delete [] indices;
            assert(err==0);
            err = gpuMesh->setVertexData( rc<float*>(mesh->mVertices), mesh->mNumVertices, 3, VERTEX_DATA_POSITION );
            assert(err==0);
            if ( mesh->HasNormals() )
            {
               err = gpuMesh->setVertexData( rc<float*>(mesh->mNormals), mesh->mNumVertices, 3, VERTEX_DATA_NORMAL );
               assert(err==0);
            }
            float* uv = new float[mesh->mNumVertices*2];
            for ( u32 k=0; k<2; k++ )
            {
                if ( mesh->HasTextureCoords(k) )
                {
                    // deform to 2 based component
                    for ( u32 j = 0; j < mesh->mNumVertices; j++ )
                    {
                        uv[j*2]   = mesh->mTextureCoords[k][j].x;
                        uv[j*2+1] = mesh->mTextureCoords[k][j].y;
                    }
                    err = gpuMesh->setVertexData(uv, mesh->mNumVertices, 2, VERTEX_DATA_UV1+k);
                    assert(err==0);
                }
            }
            delete[] uv;
            if ( mesh->HasTangentsAndBitangents() )
            {
                err = gpuMesh->setVertexData(rc<float*>(mesh->mTangents), mesh->mNumVertices, 3, VERTEX_DATA_TANGENT);
                assert(err==0);
                err = gpuMesh->setVertexData(rc<float*>(mesh->mBitangents), mesh->mNumVertices, 3, VERTEX_DATA_BITANGENT);
                assert(err==0);
            }
        }
        printf("\n\n");
        printf("Model: %s\n", name.c_str());
        printf("Faces: %fM\n", (float)totalFaces/(1000000));
        printf("Vertices: %fM\n", (float)totalVertices/(1000000));
        printf("bMin %.3f %.3f %.3f\n", bMin.x, bMin.y, bMin.z);
        printf("bMax %.3f %.3f %.3f\n", bMax.x, bMax.y, bMax.z);
        vec3 dt = bMax-bMin;
        printf("Size %.3f %.3f %.3f\n", dt.x, dt.y, dt.z);
        printf("\n");
        // We're done. Everything will be cleaned up by the importer destructor
        return true;
    }
}