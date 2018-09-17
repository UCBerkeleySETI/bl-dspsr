/***************************************************************************
 *
 *   Copyright (C) 2018 Willem van Straten and Natalia Primak
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"
#include "dsp/Observation.h"

bool dsp::GenericTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_nbit() == 2 && observation->get_npol() < 3;
}

//! Null constructor
dsp::GenericTwoBitCorrection::GenericTwoBitCorrection ()
  : TwoBitCorrection ("GenericTwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}

static unsigned mask_lsb = 1;

unsigned dsp::GenericTwoBitCorrection::get_output_ipol (unsigned idig) const
{
  if (input->get_npol() == 1)
    return 0;
  
  if (input->get_ndim() > 1)
    idig = idig >> 1;
  
  return idig && mask_lsb;
}

unsigned dsp::GenericTwoBitCorrection::get_output_ichan (unsigned idig) const
{
  if (input->get_nchan() == 1)
    return 0;

  if (input->get_ndim() > 1)
    idig = idig >> 1;

  if (input->get_npol() > 1)
    idig = idig >> 1;

  return idig;
}

unsigned dsp::GenericTwoBitCorrection::get_input_offset (unsigned idig) const
{
  return idig;
}

unsigned dsp::GenericTwoBitCorrection::get_input_incr () const
{
  return input->get_nchan() * input->get_npol() * input->get_ndim();
}

/*! The quadrature components must be offset by one */
unsigned dsp::GenericTwoBitCorrection::get_output_offset (unsigned idig) const
{
  if (input->get_ndim() == 1)
    return 0;
  else
    return idig && mask_lsb;
}

/*! The in-phase and quadrature components must be interleaved */
unsigned dsp::GenericTwoBitCorrection::get_output_incr () const
{
  return input->get_ndim();
}

