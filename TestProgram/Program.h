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

        SDL_Window*   m_window;
        SDL_Renderer* m_renderer;
        void* m_glContext;

        sptr<IScene>   m_scene;
        sptr<ICamera>  m_camera;
        sptr<Gradient> m_gradient;
        sptr<Blob>     m_blob;

        // using openGL interop
        GLTextureBufferRenderer m_glRenderer;
        GLTextureBufferObject m_textureBufferObject;

        u32* m_tempBuff;
    };
}