#pragma once
#include "Types.h"

namespace Beam
{
    class DeviceBuffer;

    class Gradient
    {
    public:
        void setup(u32 w, u32 h);
        void run();
        void transfer(u32* buffer);

    private:
        u32 m_w, m_h;
        sptr<DeviceBuffer> m_buffer;
    };
}