#include "DeviceBuffer.h"
#include "CudaHelp.h"


namespace Beam
{
    DeviceBuffer::DeviceBuffer(u32 size):
        m_size(size),
        m_devData(nullptr)
    {
        assert(m_size);
    #if CUDA
        CUDA_CALL(cudaMalloc(&m_devData, size));
    #else
        m_devData = malloc(size);
    #endif
    }

    DeviceBuffer::~DeviceBuffer()
    {
    #if CUDA
        CUDA_CALL(cudaFree(m_devData));
    #else
        free(m_devData);
    #endif
    }

    void DeviceBuffer::copyTo(void* buffer, bool wait)
    {
    #if CUDA
        if ( wait )
        {
            CUDA_CALL(cudaMemcpy(buffer, m_devData, m_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
        else
        {
            CUDA_CALL(cudaMemcpyAsync(buffer, m_devData, m_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
    #else
        memcpy( buffer, m_devData, m_size );
    #endif
    }

    void DeviceBuffer::copyFrom(const void* buffer, bool wait)
    {
    #if CUDA
        if ( wait )
        {
            CUDA_CALL(cudaMemcpy(m_devData, buffer, m_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }
        else
        {
            CUDA_CALL(cudaMemcpyAsync(m_devData, buffer, m_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }
    #else
        memcpy( m_devData, buffer, m_size );
    #endif
    }

}