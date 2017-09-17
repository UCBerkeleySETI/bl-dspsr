//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Convolution.h

#ifndef __Convolution_h
#define __Convolution_h

#include "dsp/Response.h"

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "FTransformAgent.h"

namespace dsp {
  
  class Apodization;

  //! Convolves a TimeSeries using a frequency response function
  /*! This class implements the overlap-save method of discrete
    convolution with a finite impulse response (FIR) function, as
    described in Chapter 13.1 of Numerical Recipes.

    The algorithm can perform both scalar and matrix convolution
    methods, and is highly suited to phase-coherent dispersion removal
    and phase-coherent polarimetric calibration.
    
    If g(t) is the finite impulse response function with which the
    data stream will be convolved, then the Convolution::response
    attribute represents G(w), the FFT of g(t).  Convolution::response
    may contain an array of filters, one for each frequency channel.
    
    In order to improve the spectral leakage characteristics, an
    apodization function may be applied to the data in the time domain
    by setting the Convolution::apodization attribute.
    
    Referring to Figure 13.1.3 in Numerical Recipes,
    \f$m_+\f$=response->get_impulse_pos() and
    \f$m_-\f$=response->get_impulse_neg(), so that the duration,
    M=\f$m_+ + m_-\f$, of g(t) corresponds to the number of complex
    time samples in the result of each backward FFT that are polluted
    by the cyclical convolution transformation.  Consequently,
    \f$m_+\f$ and \f$m_-\f$ complex samples are dropped from the
    beginning and end, respectively, of the result of each backward
    FFT; neighbouring FFTs will overlap by the appropriate number of
    points to make up for this loss.  */

  class Convolution: public Transformation <TimeSeries, TimeSeries> {

  public:

    //! Null constructor
    Convolution (const char* name = "Convolution", Behaviour type = outofplace);

    //! Destructor
    virtual ~Convolution ();

    //! Prepare all relevant attributes
    void prepare ();

    //! Reserve the maximum amount of output space required
    void reserve ();

    //! Get the minimum number of samples required for operation
    uint64_t get_minimum_samples () { return nsamp_fft; }

    //! Get the minimum number of samples lost
    uint64_t get_minimum_samples_lost () { return nsamp_overlap; }

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the frequency response function
    virtual void set_response (Response* response);

    //! Set the apodization function
    virtual void set_apodization (Apodization* function);

    //! Set the passband integrator
    virtual void set_passband (Response* passband);

    //! Return true if the response attribute has been set
    bool has_response () const;

    //! Return a pointer to the frequency response function
    virtual const Response* get_response() const;

    //! Return true if the passband attribute has been set
    bool has_passband () const;

    //! Return a pointer to the integrated passband
    virtual const Response* get_passband() const;

    //! Set the memory allocator to be used
    void set_device (Memory *);

    //! Engine used to perform discrete convolution step
    class Engine;

    void set_engine (Engine*);

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Frequency response (convolution kernel)
    Reference::To<Response> response;

    //! Apodization function (time domain window)
    Reference::To<Apodization> apodization;

    //! Integrated passband
    Reference::To<Response> passband;

    //! Prepare the output TimeSeries
    void prepare_output ();

  private:

    friend class Filterbank;
    friend class TFPFilterbank;
    friend class SKFilterbank;

    Reference::To<Memory> memory;

    unsigned nfilt_tot;
    unsigned nfilt_pos;
    unsigned nfilt_neg;

    unsigned nsamp_overlap;
    unsigned nsamp_step;
    unsigned nsamp_fft;

    double scalefac;

    bool matrix_convolution;

    FTransform::Plan* forward;
    FTransform::Plan* backward;

    unsigned scratch_needed;
    uint64_t npart;
    unsigned n_fft;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;
  };

  class Convolution::Engine : public Reference::Able
  {
    public:

      virtual void set_scratch (void *) = 0;

      virtual void prepare (dsp::Convolution * convolution) = 0;

      virtual void perform (const TimeSeries* in, TimeSeries* out, unsigned npart) = 0;
  };

  
}

#endif
