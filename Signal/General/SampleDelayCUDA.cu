/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SampleDelayCUDA.h"
#include "dsp/SampleDelayFunction.h"
#include "dsp/MemoryCUDA.h"
#include "Error.h"

#include <iostream>

using namespace std;

CUDA::SampleDelayEngine::SampleDelayEngine(cudaStream_t _stream)
{
  stream = _stream;
}


void CUDA::SampleDelayEngine::fpt_copy (
    const dsp::TimeSeries * input, dsp::TimeSeries * output,
    uint64_t zero_delay, uint64_t output_nfloat,
    dsp::SampleDelayFunction* function)
{

  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  if ( output -> get_memory() -> on_host () )
    kind = cudaMemcpyDeviceToHost;

  //if (stream)
  //  cudaStreamSynchronize(stream);
  //else
  //  cudaDeviceSynchronize();

  for (unsigned ipol=0; ipol < input->get_npol (); ipol++)
  {
    for (unsigned ichan=0; ichan < input->get_nchan (); ichan++)
    {
      int64_t applied_delay = 0;

      if (zero_delay)
        // delays are relative to maximum delay
        applied_delay = zero_delay - function->get_delay (ichan, ipol);
      else
        // delays are absolute and guaranteed positive
        applied_delay = function->get_delay (ichan, ipol);

      const float* in_data = input->get_datptr (ichan, ipol);
      float* out_data = output->get_datptr (ichan, ipol);

      cudaError error = cudaMemcpyAsync (
          out_data, in_data + applied_delay*input->get_ndim (), 
          output_nfloat*sizeof(float), kind, stream);

      if (error != cudaSuccess)
        throw Error (InvalidState, "CUDA::SampleDelayEngine::fpt_copy",
                     cudaGetErrorString (error));

    }
  }

  if (kind == cudaMemcpyDeviceToHost)
    cudaDeviceSynchronize();
}

