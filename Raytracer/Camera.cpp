#include "Camera.h"
#include "Scene.h"
#include "Types.h"
#include "SharedTypes.h"
#include "RenderTarget.h"
#include "Util.h"
#include <cassert>
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
using namespace std;
using namespace glm;
using namespace Beam;


extern "C"
void bmMarch(const void* root, const vec3& bMin, const vec3& bMax,
             const vec3* initialRays,
             u32* buffer, u32 pitch, u32 width, u32 height,
             const vec3& eye, const mat3& orient,
             const StaticMeshData* meshData, u32 numMeshes);

extern "C"
void bmClear(u32* buffer, u32 pitch, u32 width, u32 height, u32 clearValue);


namespace Beam
{
    // ------ ICamera ----------------------------------------------------------------------------------------

    sptr<ICamera> ICamera::create()
    {
        return make_shared<Camera>();
    }

    // ------ Camera ----------------------------------------------------------------------------------------

    Camera::Camera():
        m_width(1000),
        m_height(1000)
    {
    }

    Camera::~Camera()
    {
    }

    u32 Camera::setInitialRays(u32 width, u32 height, float left, float right, float top, float bottom, float zoom)
    {
        if ( width==0 || height==0 ) return ERROR_INVALID_PARAMETER;
        m_width  = width;
        m_height = height;
        m_initialRays   = make_shared<DeviceBuffer>(sc<u32>(sizeof(vec3))*m_width*m_height);
        if ( !m_initialRays ) return ERROR_GPU_ALLOC_FAIL;
        // setup ray buffer
        float dx = (right-left) / m_width;
        float dy = (bottom-top) / m_height;
        float ry = top + dy*.5f;
        float z2 = zoom*zoom;
        vec3* rayBuffer = new vec3[m_width*m_height];
        for ( u32 y=0; y<m_height; y++, ry += dy )
        {
            float rx = left + dx*.5f;
            for ( u32 x=0; x<m_width; x++, rx += dx )
            {
                float d = 1.f / sqrt(z2 + rx*rx + ry*ry);
                if ( ::isnan( d ) || d <= 0.f ) return ERROR_INVALID_PARAMETER;
                auto addr = y*m_width+x;
                rayBuffer[addr].x = rx*d;
                rayBuffer[addr].y = ry*d;
                rayBuffer[addr].z = zoom*d;
            }
        }
        m_initialRays->copyFrom(rayBuffer, true);
        delete [] rayBuffer;
        return ERROR_ALL_FINE;
    }

    u32 Camera::clear(sptr<IRenderTarget>& target, u32 value)
    {
        if ( !target )
        {
            return ERROR_INVALID_PARAMETER;
        }

        auto err = setRenderTarget( target );
        if ( err != ERROR_ALL_FINE ) return err;
        auto& rt = target;

        bmClear( rt->buffer<u32>(), rt->pitch(), rt->width(), rt->height(), value );
   //     cudaDeviceSynchronize();

        err = setRenderTarget( nullptr );
        if (err != ERROR_ALL_FINE ) return err;

        return ERROR_ALL_FINE;
    }

    u32 Camera::traceScene(const float* eye3, const float* orient3x3, sptr<IScene>& scene, sptr<IRenderTarget>& target)
    {
        if ( !eye3 || !orient3x3 || !scene || m_width==0 || m_height==0 ||
             !m_initialRays || !target )
        {
            return ERROR_INVALID_PARAMETER;
        }

        // upate out of date mesh ptrs (if meshes were added/removed)
        Scene* s = sc<Scene*>(scene.get());
        s->updateMeshPtrs();

        u32 err = setRenderTarget(target);
        if ( err != ERROR_ALL_FINE ) return err;
        auto& rt = target;

        if ( rt->width() != m_width || rt->height() != m_height )
        {
            return ERROR_MISMATCH_RT_DIMENSION_AND_CAMERA;
        }

        const vec3 eye      = *rc<const vec3*>(eye3);
        const mat3 orient   = *rc<const mat3*>(orient3x3);

        bmMarch
        ( 
            s->rootNode()->ptr<void>(),
            s->min(), s->max(),
            m_initialRays->ptr<const vec3>(),
            rt->buffer<u32>(),
            rt->pitch(),
            rt->width(),
            rt->height(),
            eye, orient,
            s->staticMeshPtrs()->ptr<const StaticMeshData>(),
            s->staticMeshPtrs()->size()
        );

        cudaDeviceSynchronize();
        
        return ERROR_ALL_FINE;
    }

    u32 Camera::setRenderTarget(sptr<IRenderTarget> target)
    {
        // See if not already set, otherwise unlock previous and lock new one if target is not nullptr
        sptr<IRenderTarget> prevRt = m_prevRenderTarget.lock();
        if ( prevRt != target )
        {
            if ( prevRt ) sc<RenderTarget*>(prevRt.get())->unlock();
            m_prevRenderTarget = target;
            if ( target )
            {
                void* devPtr;
                u64 size;
                u32 err = sc<RenderTarget*>(target.get())->lock(&devPtr, &size);
                if ( err != ERROR_ALL_FINE ) return err;
            }
        }
        return ERROR_ALL_FINE;
    }
}
