//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/DedispersionSampleDelay.h

#ifndef __Dedispersion_SampleDelay_h
#define __Dedispersion_SampleDelay_h

#include "dsp/Dedispersion.h"
#include "dsp/SampleDelayFunction.h"

namespace dsp {

  class Dedispersion::SampleDelay : public SampleDelayFunction {
    
  public:
    
    //! Default constructor
    SampleDelay ();
    
    //! Set up the dispersion delays
    bool match (const Observation* obs);
    
    //! Return the dispersion delay for the given frequency channel
    int64_t get_delay (unsigned ichan, unsigned ipol);
    
    //! Add to the history of operations performed on the observation
    void mark (Observation* observation);
    
  protected:
    
    //! Centre frequency of the band-limited signal in MHz
    double centre_frequency;
    
    //! Bandwidth of signal in MHz
    double bandwidth;
    
    //! Dispersion measure (in \f${\rm pc cm}^{-3}\f$)
    double dispersion_measure;
    
    //! The sampling rate (in Hz)
    double sampling_rate;
    
    //! The dispersive delays
    std::vector<int64_t> delays;
    
  };

}

#endif
