/*

 */

#ifndef __dsp_CASPSRUnpackerCUDA_h
#define __dsp_CASPSRUnpackerCUDA_h

#include<stdint.h>
#include<cuda_runtime.h>

void caspsr_texture_alloc (void * d_staging, size_t size);

void caspsr_unpack (cudaStream_t stream, const uint64_t ndat,
		    float scale,
		    const unsigned char* from,
		    float* pol0, float* pol1,
        int nthread);

#endif
