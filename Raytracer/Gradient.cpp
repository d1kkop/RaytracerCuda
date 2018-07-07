#include "Gradient.h"
#include "DeviceBuffer.h"
#include "Util.h"
using namespace std;
using namespace Beam;


extern "C"
void bmStartColorGradient(u32* buffer, u32 w, u32 h);


namespace Beam
{
    void Gradient::setup(u32 w, u32 h)
    {
        m_w = w;
        m_h = h;
        m_buffer = make_shared<DeviceBuffer>(w*h*4);
    }

    void Gradient::run()
    {
        bmStartColorGradient((m_buffer->ptr<u32>()), m_w, m_h);
    }

    void Gradient::transfer(u32* dstBuffer)
    {
        m_buffer->copyTo(dstBuffer, true);
    }
}