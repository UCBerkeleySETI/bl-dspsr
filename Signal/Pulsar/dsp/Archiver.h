//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Pulsar/dsp/Archiver.h


#ifndef __Archiver_h
#define __Archiver_h

#include "dsp/PhaseSeriesUnloader.h"
#include "Pulsar/Archive.h"

namespace Pulsar
{
  class Interpreter;
  class Integration;
  class Profile;

  class MoreProfiles;
  class dspReduction;
  class TwoBitStats;
  class DigitiserCounts;
  class Passband;
  class Backend;
}

namespace dsp
{
  class Response;
  class Operation;
  class ExcisionUnpacker;
  class HistUnpacker;

  //! Class to unload PhaseSeries data in a Pulsar::Archive
  /*! 

  */
  class Archiver : public PhaseSeriesUnloader {

  public:

    //! Verbose flag
    static unsigned verbose;

    //! Constructor
    Archiver ();
    
    //! Copy constructor
    Archiver (const Archiver&);
    
    //! Destructor
    virtual ~Archiver ();

    //! Clone operator
    Archiver* clone () const;

    //! Set the name of the Pulsar::Archive class used to create new instances
    void set_archive_class (const std::string& archive_class_name);
    void set_force_archive_class (bool);

    //! Set the post-processing script
    void set_script (const std::vector<std::string>& jobs);

    //! Set the Pulsar::Archive instance to which data will be added
    void set_archive (Pulsar::Archive* archive);

    //! Set the minimum integration length required to unload data
    void set_minimum_integration_length (double seconds);

    //! Get the Pulsar::Archive instance to which all data were added
    Pulsar::Archive* get_archive ();

    //! Add a Pulsar::Archive::Extension to those added to the output archive
    void add_extension (Pulsar::Archive::Extension* extension);

    //! Unloads all available data to a Pulsar::Archive instance
    void unload (const PhaseSeries*);

    //! Perform any clean up tasks before completion
    void finish ();

    //! Get the effective number of polarizations in the output archive
    unsigned get_npol (const PhaseSeries* phase) const;

    //! Set the Pulsar::Archive with the PhaseSeries data
    void set (Pulsar::Archive* archive, const PhaseSeries* phase);

    //! Add the PhaseSeries data to the Pulsar::Archive instance
    void add (Pulsar::Archive* archive, const PhaseSeries* phase);

    void set_archive_dedispersed (bool _archive_dedispersed)
    { archive_dedispersed = _archive_dedispersed; }
    
    bool get_archive_dedispersed() const
    { return archive_dedispersed; }
 
    //! A dspReduction extension is added to the archive with this string
    void set_archive_software(std::string _archive_software)
    { archive_software = _archive_software; }

    //! A dspReduction extension is added to the archive with this string
    std::string get_archive_software()
    { return archive_software; }

    void set_store_dynamic_extensions (bool flag)
    { store_dynamic_extensions = flag; }

    void set_use_single_archive (bool flag)
    { use_single_archive = flag; }

    void set_subints_per_file (unsigned nsub)
    { subints_per_file = nsub; }

  protected:
    
    //! Minimum integration length required to unload data
    double minimum_integration_length;

    //! Name of the Pulsar::Archive class used to create new instances
    std::string archive_class_name;

    //! do not allow the Input class to dicate the output archive file format
    bool force_archive_class;

    //! Store all output in a single archive
    bool use_single_archive;

    //! The Pulsar::Archive instance to which data will be added
    Reference::To<Pulsar::Archive> single_archive;

    //! Number of subints per output Archive (0 implies no limit)
    unsigned subints_per_file;

    //! Commands used to process Archive data before unloading
    std::vector<std::string> script;

    //! The script interpreter used to process Archive data before unloading
    Reference::To<Pulsar::Interpreter> interpreter;

    //! Response from which Passband Extension will be constructed
    Reference::To<const Response> passband;

    //! ExcisionUnpacker from which TwoBitStats Extension will be constructed
    Reference::To<const ExcisionUnpacker> excision_unpacker;

    //! HistUnpacker from which DigitiserCounts Extension will be constructed
    Reference::To<const HistUnpacker> hist_unpacker;

    //! The Pulsar::Archive::Extension classes to be added to the output
    std::vector< Reference::To<Pulsar::Archive::Extension> > extensions;

    //! Output dynamic header information (mostly diagnostic statistics)
    bool store_dynamic_extensions;

    //! Set the Pulsar::Integration with the PhaseSeries data
    void set (Pulsar::Integration* integration, const PhaseSeries* phase,
	      unsigned isub=0, unsigned nsub=1);

    //! Set the Pulsar::Profile with the specified subset of PhaseSeries data
    void set (Pulsar::Profile* profile, const PhaseSeries* phase, double scale,
	      unsigned ichan, unsigned ipol, unsigned idim);

    //! Set the Pulsar::Backend Extension
    void set (Pulsar::Backend*);

    //! Set the Pulsar::dspReduction Extension
    void set (Pulsar::dspReduction*);

    //! Set the Pulsar::dspReduction Extension
    void pack (Pulsar::dspReduction*, Operation*);

    void set_coherent_dedispersion (Signal::State state,
				    const Response* response);

    //! Set the Pulsar::TwoBitStats Extension with the dsp::TwoBitCorrection
    void set (Pulsar::TwoBitStats*);

    //! Set the Pulsar::DigitiserCounts Extension with the dsp::HistUnpacker
    void set (Pulsar::DigitiserCounts* dig_cnts, unsigned isub=0);

    //! Set the Pulsar::Passband Extension with the dsp::Response
    void set (Pulsar::Passband*);

    //! Convert raw moments to central moments of means
    void raw_to_central (unsigned ichan,
			 Pulsar::MoreProfiles* moments,
			 const Pulsar::Integration* means,
			 const unsigned* hits);

    //! Generate a new Archive for output
    Pulsar::Archive* new_Archive() const;

  private:

    //! Data are already dedispersed [default: false]
    bool archive_dedispersed;

    //! String to go in the dspReduction Extension of output archive
    std::string archive_software;

    //! Used only internally
    Reference::To<const PhaseSeries> profiles;

    //! The Pulsar::Archive instance to which data will be unloaded
    Reference::To<Pulsar::Archive> archive;

    //! Count of the corrupted profiles in a sub-integration
    unsigned corrupted_profiles;

    //! Number of fourth moments contained in output PhaseSeries
    mutable unsigned fourth_moments;
  };

}

#endif // !defined(__Archiver_h)


