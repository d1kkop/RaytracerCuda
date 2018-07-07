#pragma once
#include "Beam.h"
#include "CudaHelp.h"

namespace Beam
{
    class RenderTarget: public IRenderTarget
    {
    public:
        RenderTarget(u32 width, u32 height, u32 pitch);
        ~RenderTarget();

        u32 lock(void** devPtr, u64* size);
        u32 unlock();

        void* buffer() const override { return m_buffer; }
        u32 width() const override  { return m_width; }
        u32 height() const override { return m_height; }
        u32 pitch() const override  { return m_pitch; }

        cudaGraphicsResource* m_cudaGraphicsRT;

    private:
        u32 m_pitch, m_width, m_height;
        void* m_buffer;
    };
}