#include "RenderTarget.h"
#include "Util.h"
#include "../3rdParty/glatter-master/include/glatter/glatter.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_gl_interop.h>
using namespace std;
using namespace Beam;


namespace Beam
{
    // ----- IRenderTarget ------------------------------------------------------------------------------------------------------

    sptr<IRenderTarget> IRenderTarget::registerOpenGLRT(u32 openGLRTId, u32 width, u32 height, u32 pitch)
    {
        if ( width==0 || openGLRTId==0 || pitch<width*4 ) return nullptr;
        auto rt = make_shared<RenderTarget>(width, height, pitch);
        CUDA_CALL( cudaGraphicsGLRegisterBuffer( &rt->m_cudaGraphicsRT, openGLRTId, cudaGraphicsMapFlagsNone ) );
        //CUDA_CALL( cudaGraphicsGLRegisterImage( &rt->m_cudaGraphicsRT, openGLRTId, GL_TEXTURE_BUFFER, cudaGraphicsMapFlagsNone ) );
        return rt;
    }


    // ----- RenderTarget ------------------------------------------------------------------------------------------------------

    RenderTarget::RenderTarget(u32 width, u32 height, u32 pitch):
        m_buffer(nullptr),
        m_width(width),
        m_height(height),
        m_pitch(pitch),
        m_cudaGraphicsRT(nullptr)
    {
    }

    RenderTarget::~RenderTarget()
    {
        if ( m_buffer ) unlock();
        if ( m_cudaGraphicsRT ) CUDA_CALL( cudaGraphicsUnregisterResource( m_cudaGraphicsRT ) );
    }

    u32 RenderTarget::lock(void** devPtr, u64* size)
    {
        if ( !devPtr || !size ) return ERROR_INVALID_PARAMETER;
        CUDA_CALL( cudaGraphicsMapResources( 1, &m_cudaGraphicsRT ) );
        CUDA_CALL( cudaGraphicsResourceGetMappedPointer( devPtr, size, m_cudaGraphicsRT ) );
        m_buffer = *devPtr;
        return ERROR_ALL_FINE;
    }

    u32 RenderTarget::unlock()
    {
        CUDA_CALL( cudaGraphicsUnmapResources( 1, &m_cudaGraphicsRT ) );
        m_buffer = nullptr;
        return ERROR_ALL_FINE;
    }
}
