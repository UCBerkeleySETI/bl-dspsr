#include <stdio.h>

#include "dsp/Mark4TwoBitCorrection.h"
#include "dsp/Observation.h"
#include "dsp/TwoBitTable.h"

bool dsp::Mark4TwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "Mark4"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::Mark4TwoBitCorrection::Mark4TwoBitCorrection ()
  : TwoBitCorrection ("Mark4TwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}