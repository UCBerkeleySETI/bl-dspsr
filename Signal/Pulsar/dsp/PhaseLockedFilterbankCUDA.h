//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PhaseLockedFilterbankEngine_h
#define __PhaseLockedFilterbankEngine_h

#include "dsp/PhaseLockedFilterbank.h"
#include "dsp/TransferPhaseSeriesCUDA.h"
#include "dsp/Scratch.h"

#include <cuda_runtime.h>
#include <cufft.h>

namespace CUDA
{
  class PhaseLockedFilterbankEngine : public dsp::PhaseLockedFilterbank::Engine
  {
  public:

    PhaseLockedFilterbankEngine (cudaStream_t stream);

    //~PhaseLockedFilterbankEngine ();

    void prepare (const dsp::TimeSeries* in, unsigned nchan);

    //! sync internal PhaseSeries properties
    void setup_phase_series (dsp::PhaseSeries* output);

    //! sync data from internal PhaseSeries
    void sync_phase_series (dsp::PhaseSeries* output);

    void fpt_process (const dsp::TimeSeries* in,
        unsigned nblock, unsigned block_advance,
        unsigned phase_bin, unsigned idat_start,
        double* total_integrated);

  protected:

    cudaStream_t stream;

    //! FFT plan for spectra
    cufftHandle fft_plan;

    //! number of samples per fft
    unsigned ndat_fft;

    // memory for profiles on device
    Reference::To<dsp::PhaseSeries> m_prof;
    
    // operation used to transfer data from device to host
    Reference::To<dsp::TransferPhaseSeriesCUDA> transfer;

    // make scratch memory for FFTs
    Reference::To<dsp::Scratch> scratch;

    // pointer to scratch space
    float2* mem;

  };
}

#endif
