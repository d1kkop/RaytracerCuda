#pragma once

#include "Types.h"


namespace Beam
{
    constexpr u32 ERROR_ALL_FINE            = 0;
    constexpr u32 ERROR_NO_VERTICES         = 1;
    constexpr u32 ERROR_INVALID_PARAMETER   = 2;
    constexpr u32 ERROR_GPU_ALLOC_FAIL      = 3;
    constexpr u32 ERROR_INVALID_FORMAT      = 4;
    constexpr u32 ERROR_RT_CAM_MISMATCH     = 5;
    constexpr u32 ERROR_UNLOCK_FIRST        = 6;
    constexpr u32 ERROR_LOCK_FIRST          = 7;
    constexpr u32 ERROR_NO_RENDER_TARGET    = 8;


    constexpr u32 VERTEX_DATA_POSITION      = 0;
    constexpr u32 VERTEX_DATA_NORMAL        = 1;
    constexpr u32 VERTEX_DATA_UV1           = 2;
    constexpr u32 VERTEX_DATA_UV2           = 3;
    constexpr u32 VERTEX_DATA_TANGENT       = 4;
    constexpr u32 VERTEX_DATA_BITANGENT     = 5;
    constexpr u32 VERTEX_DATA_EXTRA1        = 6;
    constexpr u32 VERTEX_DATA_EXTRA2        = 7;
    constexpr u32 VERTEX_DATA_EXTRA3        = 8;
    constexpr u32 VERTEX_DATA_EXTRA4        = 9;
    constexpr u32 VERTEX_DATA_COUNT         = 10; // This is not a slot


    class IRenderTarget
    {
    public:
        // Register OpenGL TextureBufferObject
        static sptr<IRenderTarget> registerGLTBO(u32 rtId, u32 width, u32 height, u32 pitch);
        virtual void* buffer() const = 0;
        virtual u32 pitch() const  = 0;
        virtual u32 width() const  = 0;
        virtual u32 height() const = 0;
        virtual u32 lock() = 0;
        virtual u32 unlock() = 0;

        template <class T> T* buffer() const { return reinterpret_cast<T*>(buffer()); }
    };

    class IMesh
    {
    public:
        static sptr<IMesh> create();
        virtual u32 setVertexData(const float* vertices, u32 numVertices, u32 numComponents, u32 slotId, bool asyncCopy=false) = 0;
        virtual u32 setIndices(const u32* indices, u32 numIndices, bool asyncCopy=false) = 0;
        virtual wptr<class IScene> scene() const = 0;
    };

    class IScene
    {
    public:
        static sptr<IScene> create();
        virtual void addMesh(const sptr<IMesh>& mesh) = 0;
        virtual void removeMesh(const IMesh& mesh) = 0;
        virtual void updateGPUScene() = 0;
    };

    class ICamera
    {
    public:
        static sptr<ICamera> create();
        virtual u32 setInitialRays(u32 width, u32 height, float left=-1, float right=1, float top=1, float bottom=-1, float zoom=1) = 0;
        virtual u32 clear(u32 value) = 0;
        virtual u32 traceScene(const float* eye3, const float* orient3x3, sptr<IScene>& scene) = 0;
    };

}