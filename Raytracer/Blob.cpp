#include "Blob.h"
#include "Util.h"
#include "DeviceBuffer.h"
using namespace std;
using namespace Beam;


extern "C"
void bmStartBlob(u32* buffer, u32 w, u32 h, float time);


namespace Beam
{
    void Blob::setup(u32 w, u32 h)
    {
        m_buffer = make_shared<DeviceBuffer>(w*h*4);
        m_w = w;
        m_h = h;
    }

    void Blob::run(float time)
    {
        bmStartBlob((m_buffer->ptr<u32>()), m_w, m_h, time);
    }

    void Blob::transfer(u32* dstBuffer)
    {
        m_buffer->copyTo(dstBuffer, true);
    }
}