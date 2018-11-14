/***************************************************************************
 *
 *   Copyright (C) 2003-2018 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <assert.h>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>

#include "Pulsar/Parameters.h"
#include "T2Generator.h"

#include "dsp/File.h"
#include "dsp/Input.h"
#include "dsp/Fold.h"

#include "load_factory.h"

using namespace std;

void usage()
{
  cerr <<
    "t2pred - generate Tempo2 predictor suitable for dspsr\n"
    "  Usage: t2pred ephemeris_file input_file\n"
    "\n"
    "   -f ncoeff     number of frequency coefficients\n"
    "   -h            display help\n"
    "   -k telescope  set the telescope\n"
    "   -t ncoeff     number of time coefficients in the generator\n"
    "   -v            enable verbosity flags\n"
    "\n"
       << endl;
}

bool verbose = false;

int main(int argc, char ** argv) try
{
  bool quiet = false;

  bool verbose = false;

  string telescope;

  int ntime_coeff = -1;

  int nfreq_coeff = -1;

  int c;
  while ((c = getopt(argc, argv, "f:hk:t:v")) != -1)
    switch (c) {

    case 'f':
      nfreq_coeff = atoi (optarg);
      break;

    case 'h':
      usage ();
      return 0;

    case 'k':
      telescope.assign(optarg);
      break;

    case 't':
      ntime_coeff = atoi (optarg);
      break;

    case 'v':
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  if (optind+1 >= argc)
  {
    cerr << "t2pred: please specify file name" << endl;
    return -1;
  }

  if (verbose)
    cerr << "t2pred: loading ephemeris file " << argv[optind] << endl;

  // read the pulsar ephermeris from an ephemeris file
  Pulsar::Parameters* params = factory<Pulsar::Parameters> ( argv[optind] );

  if (verbose)
    cerr << "t2pred: opening file " << argv[optind+1] << endl;
  // read the input file 
  dsp::Input* input = dsp::File::create ( argv[optind+1] );

  // create an observation instance to access meta-data
  dsp::Observation* observation = input->get_info();

  if (telescope.size())
    observation->set_telescope (telescope);

  MJD time = observation->get_start_time()-0.01;
  Pulsar::Generator* generator = Pulsar::Generator::factory (params);
  Tempo2::Generator* t2generator = dynamic_cast<Tempo2::Generator*>(generator);

  if ((nfreq_coeff > 0 || (ntime_coeff > 0)) && !t2generator)
    throw Error (InvalidState, "t2pred", "Tempo2 ephermeris require for time or freq coeffs");

  /*
   * Tempo2 predictor code:
   *
   * Here we make a predictor valid for the next 24 hours
   * I consider this to be a bit of a hack, since theoreticaly
   * observations could be longer, and it's a bit silly to make
   * such a polyco for a 10 min obs.
   *
   */
  MJD endtime = time + 86400;

  generator->set_site( observation->get_telescope() );
  generator->set_parameters( params );
  generator->set_time_span( time, endtime );

  double freq = observation->get_centre_frequency();
  double bw = fabs( observation->get_bandwidth() );
  generator->set_frequency_span( freq-bw/2, freq+bw/2 );

  if (t2generator)
  {
    if (ntime_coeff > 0)
      t2generator->set_time_ncoeff( ntime_coeff );
    if (nfreq_coeff > 0)
      t2generator->set_frequency_ncoeff( nfreq_coeff );
  }

  // this will generate the predictor and should create the necessary files
  generator->generate ();
 
  return 0;

}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

