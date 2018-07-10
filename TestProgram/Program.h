#pragma once

#include "../Raytracer/Beam.h"
#include "../Raytracer/Gradient.h"
#include "../Raytracer/Blob.h"
#include "../Raytracer/Util.h"
#include "../Raytracer/GLinterop.h"
#include <string>

struct SDL_Window;
struct SDL_Renderer;

using namespace Beam;


namespace TestProgram
{
    struct ProfileItem
    {
        std::string name;
        double start;
        double end;

        ProfileItem(const std::string& s):
            name(s),
            start(Util::timeD())
        {
        }
    };

    struct ProfileItemWithCuda : public ProfileItem
    {
        //cudaEvent_t cstart, cend;

        //ProfileItemWithCuda(const std::string& s):
        //    cstart(nullptr),
        //    cend(nullptr),
        //    ProfileItem(s)
        //{
        //    CUDA_CALL( cudaEventCreate( &cstart ) );
        //}

        //~ProfileItemWithCuda()
        //{
        //    CUDA_CALL( cudaEventDestroy( cstart ) );
        //    CUDA_CALL( cudaEventDestroy( cend ) );
        //}
    };

    class Program
    {
    public:
        Program();
        ~Program();

        void initialize(const std::string& name, u32 width, u32 height);
        void run();

    private:
        void initSDL(const std::string& name, u32 width, u32 height);
        void closeSDL();
        void renderFrame();
        void printProfileItems();
        void pushProfile(ProfileItem& pi);

        SDL_Window*   m_window;
        SDL_Renderer* m_renderer;
        void* m_glContext;

        sptr<IScene>   m_scene;
        sptr<ICamera>  m_camera;
        sptr<Gradient> m_gradient;
        sptr<Blob>     m_blob;

        // using openGL interop
        GLTextureBufferRenderer m_glRenderer;
        GLTextureBufferObject m_textureBufferObject[4];
        u32 m_bufferObjIdx;
        
        std::vector<ProfileItem> m_profileItems;
    };
}