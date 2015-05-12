//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_dsp_PolnReshape_h
#define __baseband_dsp_PolnReshape_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

// Detection is handled very efficiently on the GPU for 2pol, analytic data.
// But other formats may be more useful down the signal path.  This
// Transformation allows post-detection conversion from 2pol,2dim data to
// a variety of formats:

// npol=4, ndim = 1: Coherence / Stokes
// npol=2, ndim = 1: PPQQ
// npol=1, ndim = 1: Intensity

// The transformation performed is determined uniquely by the output state.

namespace dsp
{
  //! Convert npol=2,ndim=2 format to variety of other data shapes
  class PolnReshape : public Transformation<TimeSeries,TimeSeries>
  {

  public:

    //! Default constructor
    PolnReshape ();

    //! Apply the npol to single-pol transformation
    void transformation ();

    //! Reshape the poln index to keep
    void set_state ( Signal::State _state) { state = _state; }

  protected:

    //! The polarization to keep
    Signal::State state;

    //! Handle 2x2 --> 4x1 (Coherence or Stokes)
    void npol4_ndim1();

    //! Handle 2x2 --> 2x1 (PPQQ)
    void npol2_ndim1();

    //! Handle 2x2 --> 1x1 (Intensity)
    void npol1_ndim1();

  };
}

#endif
