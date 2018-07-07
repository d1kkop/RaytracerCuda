#pragma once
#include "Types.h"

namespace Beam
{
    class DeviceBuffer;

    class Blob
    {
    public:
        void setup(u32 w, u32 h);
        void run(float time);
        void transfer(u32* buffer);

    private:
        u32 m_w, m_h;
        sptr<DeviceBuffer> m_buffer;
    };
}