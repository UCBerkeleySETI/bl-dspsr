//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Andrew JAmeson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_UWBFloatUnpacker_h
#define __baseband_cuda_UWBFloatUnpacker_h

#include "dsp/UWBFloatUnpacker.h"
#include <cuda_runtime.h>

namespace CUDA
{
  class UWBFloatUnpackerEngine : public dsp::UWBFloatUnpacker::Engine
  {
  public:

    //! Default Constructor
    UWBFloatUnpackerEngine (cudaStream_t stream);

    void setup ();

    bool get_device_supported (dsp::Memory* memory) const;

    void set_device (dsp::Memory* memory);

    void unpack (const dsp::BitSeries * input, dsp::TimeSeries * output);

  protected:

    cudaStream_t stream;

    struct cudaDeviceProp gpu;

    dsp::BitSeries staging;

  };
}

#endif
