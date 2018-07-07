#pragma once
#include "Beam.h"
#include "DeviceBuffer.h"

namespace Beam
{
    class Camera: public ICamera
    {
    public:
        Camera();
        ~Camera();

        u32 setInitialRays(u32 width, u32 height, float left, float right, float top, float bottom, float zoom) override;
        u32 clear(sptr<IRenderTarget>& target, u32 value) override;
        u32 traceScene(const float* eye3, const float* orient3x3, sptr<IScene>& scene, sptr<IRenderTarget>& target) override;

    private:
        u32 setRenderTarget( sptr<IRenderTarget> target );

        u32 m_width, m_height;
        sptr<DeviceBuffer>  m_initialRays;
        wptr<IRenderTarget> m_prevRenderTarget;
    };
}