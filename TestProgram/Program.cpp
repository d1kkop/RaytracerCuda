#include "Program.h"
#include "Model.h"
#include <iostream>
#include <cassert>
#include <SDL.h>
//#include <SDL_opengl.h>
#include "../Raytracer/CudaHelp.h"
#include "../Raytracer/GLinterop.h"
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "glm/vec3.hpp"
#include "glm/mat3x3.hpp"
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


namespace TestProgram
{
    Program::Program():
        m_window(nullptr),
        m_renderer(nullptr),
        m_glContext(nullptr),
        m_camera(nullptr),
        m_bufferObjIdx(0)
    {
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
        if ( !Model::load(R"(D:\_Programming\2018\RaytracerCuda\Content/f16.obj)", m_scene) )
        {
            cout << "Failed to load bunny" << endl;
        }

        //sptr<IMesh> mesh = IMesh::create();
        //vec3* vertices = new vec3[4];
        //vertices[0] = vec3(-1.f, -1.f, 1.f);
        //vertices[1] = vec3(0.f, 1.f, 1.f);
        //vertices[2] = vec3(1.f, -1.f, 1.f);
        //vertices[3] = vec3(1.f, 1.f, 1.f);
        //u32* indices = new u32[6];
        //indices[0]=0;
        //indices[1]=1;
        //indices[2]=2;
        //indices[3]=1;
        //indices[4]=2;
        //indices[5]=3;
        //vec4* colors = new vec4[4];
        //colors[0] = vec4(1.f, 0.f, 0.f, 1.f);
        //colors[1] = vec4(0.f, 1.f, 0.f, 1.f);
        //colors[2] = vec4(0.f, 0.f, 1.f, 1.f);
        //colors[3] = vec4(1.f, 1.f, 0.f, 1.f);
        //err = mesh->setIndices(indices, 3);
        //assert(err==0);
        //err = mesh->setVertexData((float*)vertices, 4, 3, VERTEX_DATA_POSITION);
        //assert(err==0);
        //err = mesh->setVertexData((float*)colors, 4, 4, VERTEX_DATA_EXTRA4);
        //assert(err==0);
        //delete[] colors;
        //delete[] indices;
        //delete[] vertices;
        //m_scene->addMesh(mesh);

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
            while ( SDL_PollEvent(&event) )
            {
                switch ( event.type )
                {
                case SDL_KEYDOWN:
                    if ( event.key.keysym.sym == SDLK_ESCAPE ) return;
                    break;

                case SDL_QUIT:
                    return;
                    break;
                }
            }
            pushProfile(input);

            renderFrame();
        }
    }

    void Program::initSDL(const string& name, u32 width, u32 height)
    {
        SDL_CALL(SDL_Init(SDL_INIT_VIDEO));
        u32 flags   = SDL_WINDOW_OPENGL; //  | SDL_WINDOW_FULLSCREEN;
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

        ProfileItem pigpuScene(R"(Scene)");
        m_scene->updateGPUScene();
        pushProfile(pigpuScene);

        //ProfileItem piu("Unlock");
        //m_bufferObjIdx = (m_bufferObjIdx+1)%NUM_RT;
        //err=m_textureBufferObject[m_bufferObjIdx].renderTarget()->unlock();
        //assert(err==0);
        //pushProfile(piu);

        //ProfileItem pil("Lock");
        //auto& rt = m_textureBufferObject[m_bufferObjIdx].renderTarget();
        //err = rt->lock();
        //assert(err==0);
        //pushProfile(pil);

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
            vec3 eye    = vec3(0, 0, -2.1f);
            mat3 orient = mat3(1.f);
            err = m_camera->traceScene(&eye.x, &orient[0][0], m_scene);
            assert(err==0);
        }
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