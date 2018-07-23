#pragma once
#include "Beam.h"
#include "CudaComon.cuh"

namespace Beam
{
    class RenderTarget: public IRenderTarget
    {
    public:
        RenderTarget(u32 width, u32 height, u32 pitch);
        ~RenderTarget();

        u32 lock() override;
        u32 unlock() override;

        void* buffer() const override { return m_buffer; }
        u32 width() const override  { return m_width; }
        u32 height() const override { return m_height; }
        u32 pitch() const override  { return m_pitch; }

        static IRenderTarget* get();
        union
        {
            cudaGraphicsResource* m_cudaGraphicsRT;
            u64 m_glTbo;
        };

    private:
        u32 m_pitch, m_width, m_height;
        void* m_buffer;

        static IRenderTarget* m_RT;
    };
}