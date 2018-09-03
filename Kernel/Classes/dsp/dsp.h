//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/dsp.h

#ifndef __baseband_dsp_h
#define __baseband_dsp_h

#include "Warning.h"

/*! \mainpage 
 
  \section intro Introduction
 
  DSPSR implements a family of C++ classes that may be used to load,
  transform and reduce observational data, primarily data that are
  regularly sampled in time (and, optionally, frequency).  This
  includes both phase-coherent data, as stored by baseband recording
  systems, and detected data, as produced by a filterbank system.  The
  functionality, contained in the dsp namespace, is divided into three
  main classes: data containers, operations, and auxilliary objects.

  The most general data container is the dsp::TimeSeries class, which
  is used to store the floating point representation of the signal in
  a variety of states.  The dsp::BitSeries class is used to store the
  N-bit digitized data before unpacking into a TimeSeries object.  The
  dsp::Loader class and its children are used to load data into the
  dsp::TimeSeries container.

  The main DSP algorithms are implemented by dsp::Operation and its
  sub-classes.  These operate on dsp::TimeSeries and can:
  <UL>
  <LI> convert digitized data to floating points (dsp::Unpack class)
  <LI> coherently dedisperse data (dsp::Convolution class)
  <LI> fold data using polyco (dsp::Fold class)
  <LI> etc...
  </UL>

  The auxilliary classes perform operations on arrays of data, such as
  multiplying a frequency response matrix by a spectrum field vector
  (e.g. the dsp::Response class).

  \section backend Adding a new File Format

  When adding a new file format, the following steps should be followed:
  <UL>

  <LI> Create a new subdirectory of <tt>dspsr/Kernel/Formats</tt>, 
  say <tt>dspsr/Kernel/Formats/myformat</tt>, 
  and create all new files here.

  <LI> Inherit dsp::File or one of its derived classes, implement the
  header-parsing code, and add the new class to
  <tt>dspsr/Kernel/Formats/File_registry.C</tt> using preprocessor
  directives.

  <LI> Inherit dsp::Unpacker or one of its derived classes, implement
  the bit-unpacking code, and add the new class to
  <tt>dspsr/Kernel/Formats/Unpacker_registry.C</tt> using preprocessor
  directives.

  <LI> Add the name of the new subdirectory to the ASCII text file
  named <tt>backends.list</tt> in the DSPSR build directory (where you
  type <kbd>make</kbd>).

  </UL>

 */

//! Contains all Baseband Data Reduction Library classes
namespace dsp {

  //! Set true to enable backward compatibility features
  extern bool psrdisp_compatible;

  //! The baseband/dsp version number
  extern const float version;

  //! Set the verbosity level of various base classes
  void set_verbosity (unsigned level);

  //! Warning messages filter
  extern Warning warn;

}

#endif







