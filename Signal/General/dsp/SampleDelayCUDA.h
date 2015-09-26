
//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SampleDelayEngine_h
#define __SampleDelayEngine_h

#include "dsp/SampleDelay.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class SampleDelayEngine : public dsp::SampleDelay::Engine
  {
  public:

    SampleDelayEngine (cudaStream_t stream);

    void fpt_copy (const dsp::TimeSeries* in, dsp::TimeSeries* out,
        uint64_t zero_delay, uint64_t output_nfloat,
        dsp::SampleDelayFunction* function);

  protected:

    cudaStream_t stream;

  };
}

#endif
