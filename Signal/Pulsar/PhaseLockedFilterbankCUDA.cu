/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PhaseLockedFilterbankCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "CUFFTError.h"

CUDA::PhaseLockedFilterbankEngine::PhaseLockedFilterbankEngine(
    cudaStream_t _stream)
{
  stream = _stream;
  fft_plan = 0;
  mem = 0;

  m_prof = new dsp::PhaseSeries;
  m_prof->set_memory (new CUDA::DeviceMemory (stream));

  scratch = new dsp::Scratch;
  scratch->set_memory (new CUDA::DeviceMemory (stream));
}

// TODO -- make destructor work
//CUDA::PhaseLockedFilterbankEngine::~PhaseLockedFilterbankEngine ()
//{
//  cufftDestroy (fft_plan);
//}

void CUDA::PhaseLockedFilterbankEngine::setup_phase_series (
    dsp::PhaseSeries* output)
{
  m_prof->copy_configuration (output);
  m_prof->resize (output->get_nbin());
  m_prof->zero ();
}


void CUDA::PhaseLockedFilterbankEngine::sync_phase_series (
    dsp::PhaseSeries* output)
{
  // first, sync cached version with output for metadata
  m_prof->copy_configuration (output);

  // then transfer data and metadata back over
  if (!transfer)
    transfer = new dsp::TransferPhaseSeriesCUDA(stream);
  transfer->set_kind ( cudaMemcpyDeviceToHost );
  transfer->set_input ( m_prof );
  transfer->set_output ( output );
  transfer->set_transfer_hits ( false );
  transfer->operate ();

}

// Each thread processes an output frequency channel by forming the
// coherency parameters from the two complex polarizations.
// By fiat, the layout of the input data is input_nchan x 2 x nchan,
// i.e. block x pol x frequency.
__global__ void detect (float2* in, float* out, 
    unsigned input_nchan, unsigned nchan, 
    unsigned chan_stride, unsigned poln_stride)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;    
  // total output channels = nchan * input_nchan
  if (idx >= nchan*input_nchan)
    return;

  // figure out which input channel we are in
  int inchan = idx / nchan;
  int Aidx = inchan*nchan + idx;

  // polarization blocks are contiguous
  float2 A = in[Aidx];
  float2 B = in[Aidx + nchan];

  // coherency parameters
  float AA = A.x*A.x + A.y*A.y;
  float BB = B.x*B.x + B.y*B.y;
  float reAB = A.x*B.x + A.y*B.y;
  float imAB = A.x*B.y - A.y*B.x;

  // idx is the output channel number
  out += idx * chan_stride;
  *out += AA;
  out += poln_stride;
  *out += BB;
  out += poln_stride;
  *out += reAB;
  out += poln_stride;
  *out += imAB;

}

void CUDA::PhaseLockedFilterbankEngine::prepare (
    const dsp::TimeSeries* input, unsigned nchan)
{

  ndat_fft = nchan; // always complex

  // prepare FFT plan
  cufftResult result;

  int rank = 1;
  int batch = input->get_nchan () * 2; // input nchan * npol
  int dims[1] = {nchan};
  int istride = 1;
  int idist = (input->get_datptr (0,1) - input->get_datptr (0,0))/2;
  int inembed[1] = {idist};
  int ostride = 1;
  int odist = nchan;
  int onembed[1] = {odist};

  result = cufftPlanMany (&fft_plan, rank, dims,
    inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);

  if (result != CUFFT_SUCCESS)
  {
    fft_plan = 0;
    throw CUFFTError (result, "CUDA::PhaseLockedFilterbankEngine::prepare",
          "cufftPlanMany(fft_plan, CUFFT_C2C)");
  }

  //result = cufftSetStream (fft_plan, stream);

  //if (result != CUFFT_SUCCESS)
  //{
  //  fft_plan = 0;
  //  throw CUFFTError (result, "CUDA::PhaseLockedFilterbankEngine::prepare",
  //        "cufftSetStream(fft_plan, stream)");
  //}

  // prepare scratch space
  mem = scratch->space<float2> (batch * nchan);
}


void CUDA::PhaseLockedFilterbankEngine::fpt_process (
    const dsp::TimeSeries* input,
    unsigned nblock, unsigned block_advance,
    unsigned phase_bin, unsigned idat_start, double* total_integrated)
{

  // NB npol == ndim == 2 by fiat so far
  unsigned input_nchan = input->get_nchan ();
  double time_per_sample = 1. / input->get_rate();
  double integrated_this_time = 0.;
  unsigned sample_offset = 0;
  cufftResult result;

  // compute output strides
  unsigned chan_stride = m_prof->get_datptr(1,0) - m_prof->get_datptr(0,0);
  unsigned poln_stride = m_prof->get_datptr(0,1) - m_prof->get_datptr(0,0);

  unsigned cstride = input->get_datptr (1,0) - input->get_datptr (0,0);
  unsigned pstride = input->get_datptr (0,1) - input->get_datptr (0,0);

  for (unsigned iblock = 0; iblock < nblock; iblock++)
  {
    // make somewhat arbitrary choice of how to assign integration time
    // to blocks -- by letting the first block contribute all its samples,
    // works for the case of overlap==false
    unsigned unique_samples = iblock ? block_advance : ndat_fft;

    // Update totals
    integrated_this_time += time_per_sample * unique_samples;

    // do batch FFT
    float2* fft_in_ptr = (float2*) input->get_datptr (0, 0);
    fft_in_ptr += idat_start + sample_offset;
    result = cufftExecC2C (
        fft_plan, fft_in_ptr, mem, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, 
          "CUDA::PhaseLockedFilterbankEngine::fpt_process", "cufftExecC2C");
    //cudaDeviceSynchronize ();
    //cudaStreamSynchronize (stream);

    // do detection over input channels
    int threads = 256;
    dim3 blocks;
    blocks.x = (input_nchan*ndat_fft)/threads + 1;
    float* out_ptr = m_prof->get_datptr (0, 0) + phase_bin;
    detect<<<blocks,threads>>> (mem, out_ptr, 
        input_nchan, ndat_fft, chan_stride, poln_stride);
    //cudaStreamSynchronize (stream);

    sample_offset += block_advance;

  } // for each block in time series

  // update hits
  m_prof->get_hits()[phase_bin] += nblock;
  m_prof->increment_integration_length (integrated_this_time);
  *total_integrated += integrated_this_time;
  //m_prof->ndat_total += nblock;
}
