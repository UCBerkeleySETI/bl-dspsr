//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 Willem van Straten and Natalia Primak
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/GenericTwoBitCorrection.h

#ifndef __GenericTwoBitCorrection_h
#define __GenericTwoBitCorrection_h

class GenericTwoBitCorrection;

#include "dsp/TwoBitCorrection.h"

namespace dsp {

  class TwoBitTable;

  //! Converts Generic data from 2-bit digitized to floating point values
  class GenericTwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor initializes base class attributes
    GenericTwoBitCorrection ();

    //! Return true if GenericTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_ipol (unsigned idig) const;

    unsigned get_output_ichan (unsigned idig) const;

    unsigned get_output_offset (unsigned idig) const;

    unsigned get_output_incr () const;

    unsigned get_input_offset (unsigned idig) const;

    unsigned get_input_incr () const;

  };
  
}

#endif
