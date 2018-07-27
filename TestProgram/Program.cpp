#include "Program.h"
#include "Model.h"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <SDL.h>
//#include <SDL_opengl.h>
#include "../Raytracer/GLinterop.h"
#include "../Raytracer/CudaComon.cuh"
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "glm/vec3.hpp"
#include "glm/mat3x3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
using namespace std;
using namespace glm;
using namespace Beam;

#define NUM_RT 1


#define LIB_CALL( expr, name ) \
{ \
	auto status=expr; \
	if ( status != 0 )  \
	{ \
		cout << name << " ERROR: " << (status) << endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}

#define SDL_CALL( expr ) LIB_CALL( expr, "SDL" )


bool bmGLHasContext()
{
    return SDL_GL_GetCurrentContext()!=nullptr;
}


u32 myHash(u32 h)
{
    u16 sum1 = 0;
    u16 sum2 = 0;
    int index;
    u8* data = (u8*)&h;
    for ( index = 0; index < 4; ++index )
    {
        sum1 = (sum1 + data[index]) % 255;
        sum2 = (sum2 + sum1) % 255;
    }
    h = (sum2 << 8) | sum1;
    assert( h < 65536 );
    return h;
}

float bmBoxRayIntersectNoZero(const vec3& bMin, const vec3& bMax,
                              const vec3& orig, const vec3& invDir);
float bmBoxRayIntersect(const vec3& bMin, const vec3& bMax,
                              const vec3& orig, const vec3& invDir);

namespace TestProgram
{
    Program::Program():
        m_window(nullptr),
        m_renderer(nullptr),
        m_glContext(nullptr),
        m_camera(nullptr),
        m_bufferObjIdx(0)
    {
        vec3 eye(0,0,-2.3f);
        vec3 dir(-1,-1,-1);
        dir = normalize(dir);
        vec3 invDir(1.f/dir.x,1.f/dir.y,1.f/dir.z);
        float d = bmBoxRayIntersectNoZero(
            vec3(0.f,0.f,-3.f), vec3(1.f,1.f,-2.f), 
            eye, invDir );
        float d2 = bmBoxRayIntersect( 
            vec3(0.f, 0.f, -3.f), vec3(1.f, 1.f, -2.f),
            eye, invDir );

        int j = 0;
        //int r=100;
        //vector<int> hh;
        //for ( int i = -r; i <r; i++ )
        //{
        //    int h = myHash(i);
        //    printf("i %d, h %d\n", i, h);
        //    hh.emplace_back(h);
        //}
        //std::unique(hh.begin(), hh.end());
        //printf("Unique: %zd\n", hh.size());

        //int j = 0;

        m_pos = vec3(0.f, 0.f, -2.1f);
        m_pan = 0;
        m_pitch = 0;
    }

    Program::~Program()
    {
        closeSDL();
    }

    void Program::initialize(const string& name, u32 width, u32 height)
    {
        Util::time();
        initSDL(name, width, height);

        int count=0;
        CUDA_CALL( cudaGetDeviceCount( &count ) );

        CUDA_CALL( cudaSetDevice( count-1 ) );

        bool succes; 
        succes = m_glRenderer.init(width, height);
        assert(succes);
        for ( auto& rt : m_textureBufferObject ) 
            succes = rt.init(width, height);
        assert(succes);

        //m_gradient = make_shared<Gradient>();
        //m_gradient->setup(width, height);
        //m_blob = make_shared<Blob>();
        //m_blob->setup(width, height);
        // ---- scene setup -----

        u32 err=0;
        m_scene = IScene::create();

        if ( !Model::load(R"(D:\_Programming\2018\RaytracerCuda\Content/armadillo.obj)", m_scene, 1) )
        {
            cout << "Failed to load armadillo" << endl;
        }

        if ( !Model::load(R"(D:\_Programming\2018\RaytracerCuda\Content/f16.obj)", m_scene, 1) )
        {
            cout << "Failed to load f16" << endl;
        }


    /*    sptr<IMesh> mesh = IMesh::create();
        vec3* vertices = new vec3[4];
        vec3* normals  = new vec3[4];
        vertices[0] = vec3(-1.f, -1.f, 1.56f);
        vertices[1] = vec3(0.f, 1.f, 1.56f);
        vertices[2] = vec3(1.f, -1.f, 1.56f);
        vertices[3] = vec3(2.f, 1.f, 1.56f);
        for ( int i=0; i<4; i++ ) normals[i] = vec3(0, 0, -1);
        u32* indices = new u32[6];
        indices[0]=0;
        indices[1]=1;
        indices[2]=2;
        indices[3]=1;
        indices[4]=2;
        indices[5]=3;
        vec4* colors = new vec4[4];
        colors[0] = vec4(1.f, 0.f, 0.f, 1.f);
        colors[1] = vec4(0.f, 1.f, 0.f, 1.f);
        colors[2] = vec4(0.f, 0.f, 1.f, 1.f);
        colors[3] = vec4(1.f, 1.f, 0.f, 1.f);
        err = mesh->setIndices(indices, 3);
        assert(err==0);
        err = mesh->setVertexData((float*)vertices, 4, 3, VERTEX_DATA_POSITION);
        assert(err==0);
        err = mesh->setVertexData((float*)colors, 4, 4, VERTEX_DATA_EXTRA4);
        assert(err==0);
        err = mesh->setVertexData((float*)normals, 4, 3, VERTEX_DATA_NORMAL);
        assert(err==0);
        delete[] colors;
        delete[] indices;
        delete[] vertices;
        delete[] normals;
        m_scene->addMesh(mesh);*/

//        u32 err;
        m_camera = ICamera::create();
        err = m_camera->setInitialRays( width, height );
        assert(err==0);

        err = m_textureBufferObject[0].renderTarget()->lock();
        assert(err==0);
    }

    void Program::run()
    {
        ProfileItem frame("Frame");
        while ( true )
        {
            pushProfile(frame);
            printProfileItems();
            frame.start = Util::timeD();

            // handle window events
            ProfileItem input("Poll");
            SDL_Event event;
            vec3 move(0);
            float speed = .3f;
            float mspeed = 0.004f;
            static bool kds[6] ={ false, false, false, false };
            while ( SDL_PollEvent(&event) )
            {
                switch ( event.type )
                {
                case SDL_KEYDOWN:
                    if ( event.key.keysym.sym == SDLK_ESCAPE ) return;
                    if ( event.key.keysym.sym == SDLK_a ) kds[0]=true;
                    if ( event.key.keysym.sym == SDLK_d ) kds[1]=true;
                    if ( event.key.keysym.sym == SDLK_w ) kds[2]=true;
                    if ( event.key.keysym.sym == SDLK_s ) kds[3]=true;
                    if ( event.key.keysym.sym == SDLK_q ) kds[4]=true;
                    if ( event.key.keysym.sym == SDLK_e ) kds[5]=true;
                    break;

                case SDL_KEYUP:
                    if ( event.key.keysym.sym == SDLK_a ) kds[0]=false;
                    if ( event.key.keysym.sym == SDLK_d ) kds[1]=false;
                    if ( event.key.keysym.sym == SDLK_w ) kds[2]=false;
                    if ( event.key.keysym.sym == SDLK_s ) kds[3]=false;
                    if ( event.key.keysym.sym == SDLK_q ) kds[4]=false;
                    if ( event.key.keysym.sym == SDLK_e ) kds[5]=false;
                    break;

                case SDL_MOUSEMOTION:
                    m_pan += event.motion.xrel * mspeed;
                    m_pitch -= event.motion.yrel * mspeed;
                    break;

                case SDL_QUIT:
                    return;
                    break;
                }
            }
            pushProfile(input);

            // Update 
            if ( kds[0] ) move.x -= speed;
            if ( kds[1] ) move.x += speed;
            if ( kds[2] ) move.z += speed;
            if ( kds[3] ) move.z -= speed;

            mat4 yaw   = rotate(m_pan, vec3(0.f, 1.f, 0.f));
            mat4 pitch = rotate(m_pitch, vec3(1.f, 0.f, 0.f));
            mat3 orient = (yaw * pitch);
      //      m_pos += orient*move;
            if ( kds[4] ) m_pos.y += speed;
            if ( kds[5] ) m_pos.y -= speed;

            // Draw
            renderFrame();
        }
    }

    void Program::initSDL(const string& name, u32 width, u32 height)
    {
        SDL_CALL(SDL_Init(SDL_INIT_VIDEO));
        u32 flags   = SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN;
        m_window    = SDL_CreateWindow(name.c_str(), 100, 100, width, height, flags);
        m_renderer  = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED);
        m_glContext = SDL_GL_CreateContext(m_window);
        SDL_CALL(SDL_GL_MakeCurrent(m_window, m_glContext));
    }

    void Program::closeSDL()
    {
        SDL_GL_DeleteContext(m_glContext);
        SDL_DestroyRenderer(m_renderer);
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        m_glContext=nullptr;
        m_renderer=nullptr;
        m_window=nullptr;
    }

    void Program::renderFrame()
    {
        u32 err=0;

        static bool first=true;
        if ( first )
        {
            first = false;
            ProfileItem pigpuScene(R"(Scene)");
            m_scene->updateGPUScene();
            cudaDeviceSynchronize(); // DEBUG
            pushProfile(pigpuScene);
        }

        ProfileItem piu("Unlock");
        m_bufferObjIdx = (m_bufferObjIdx+1)%NUM_RT;
        err=m_textureBufferObject[m_bufferObjIdx].renderTarget()->unlock();
        assert(err==0);
        pushProfile(piu);

        ProfileItem pil("Lock");
        auto& rt = m_textureBufferObject[m_bufferObjIdx].renderTarget();
        err = rt->lock();
        assert(err==0);
        pushProfile(pil);

        // -- Clear --
        {
//             static int c=0;
//             c = (c+1)%256;
//             ProfileItem pic("Clear");
//             err = m_camera->clear(c<<16);
//             assert(err==0);
//             pushProfile(pic);
        }

        // -- Trace scene --
        ProfileItem piTrace("Trace");
        {
            mat4 yaw   = rotate(m_pan, vec3(0.f, 1.f, 0.f));
            mat4 pitch = rotate(m_pitch, vec3(1.f, 0.f, 0.f));
            mat3 orient = mat3(1); //( yaw * pitch );
            err = m_camera->traceScene(&m_pos.x, &orient[0][0], m_scene);
            assert(err==0);
        } 
        cudaDeviceSynchronize(); // DEBUG
        pushProfile(piTrace);

      //  cudaDeviceSynchronize();


        // -- Render texture to backbuffer
        ProfileItem pir("Render");
        m_glRenderer.render(m_textureBufferObject[m_bufferObjIdx]);
        pushProfile(pir);

  /*      ProfileItem picusync("CudaSync");
        cudaDeviceSynchronize();
        pushProfile(picusync);

        ProfileItem pisync("Sync");
        m_glRenderer.sync();
        pushProfile(pisync);*/

        // -- Flip front with back buffer of displaying window --
        ProfileItem pisw("Swap");
        SDL_GL_SwapWindow( m_window );
        pushProfile(pisw);
    }


    void Program::printProfileItems()
    {
        static float t=0;
        if ( Util::time() - t >= 1.f )
        {
            t = Util::time();
            cout << "--- Profile Items ---" << endl;
            for ( auto& pi : m_profileItems )
            {
                double el = (pi.end-pi.start)*1000;
                cout << pi.name << "\t" << el << endl;
            }
            cout << endl;
        }
        m_profileItems.clear();
    }

    void Program::pushProfile(ProfileItem& pi)
    {
        pi.end = Util::timeD();
        m_profileItems.push_back( pi );
    }

}