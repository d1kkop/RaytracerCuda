#include "DeviceBuffer.h"
#include "CudaHelp.h"


namespace Beam
{
    DeviceBuffer::DeviceBuffer(u32 size):
        m_size(size),
        m_devData(nullptr)
    {
        assert(m_size);
        CUDA_CALL(cudaMalloc(&m_devData, size));
    }

    DeviceBuffer::~DeviceBuffer()
    {
        CUDA_CALL(cudaFree(m_devData));
    }

    void DeviceBuffer::copyTo(void* buffer, bool wait)
    {
        if ( wait )
        {
            CUDA_CALL(cudaMemcpy(buffer, m_devData, m_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
        else
        {
            CUDA_CALL(cudaMemcpyAsync(buffer, m_devData, m_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
    }

    void DeviceBuffer::copyFrom(const void* buffer, bool wait)
    {
        if ( wait )
        {
            CUDA_CALL(cudaMemcpy(m_devData, buffer, m_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }
        else
        {
            CUDA_CALL(cudaMemcpyAsync(m_devData, buffer, m_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }
    }

}