//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/ASCIIObservation.h

#ifndef __ASCIIObservation_h
#define __ASCIIObservation_h

#include "dsp/Observation.h"
#include "ascii_header.h"

namespace dsp {
 
  //! Parses Observation attributes from an ASCII header
  /*! This class parses the ASCII header block used by DADA-based
    instruments such as CPSR2, PuMa2, and APSR.  It initializes all of
    the attributes of the Observation base class.  The header block
    may come from a data file, or from shared memory. */
  class ASCIIObservation : public Observation {

  public:

    //! Construct from an ASCII header block
    ASCIIObservation (const char* header=0);

    //! Construct from an Observation
    ASCIIObservation (const Observation*);

    //! Read the ASCII header block
    void load (const char* header);

    //! Write an ASCII header block
    void unload (char* header);

    //! Get the number of bytes offset from the beginning of acquisition
    uint64_t get_offset_bytes () const { return offset_bytes; }

    //! Set/unset a required keyword
    void set_required (std::string key, bool required=true);

    //! Check if a certain keyword is required
    bool is_required (std::string key);

    template <typename T>
    int custom_header_get (std::string key, const char *format, T result) const;

  protected:

    std::string hdr_version;

    //! The list of ASCII keywords that must be present
    std::vector< std::string > required_keys;

    //! Load a keyword, only throw an error if it's required and doesn't exist
    template <typename T>
    int ascii_header_check (const char *header, std::string key, 
        const char *format, T result);


    //! Number of bytes offset from the beginning of acquisition
    uint64_t offset_bytes;

    std::string loaded_header;
  };
  
}

template <typename T>
int dsp::ASCIIObservation::custom_header_get ( std::string key, 
    const char *format, T result) const
{
  int rv = ascii_header_get (loaded_header.c_str(), key.c_str(), format, result);
  if ( rv > 0)
    return rv;
  throw Error (InvalidState, "ASCIIObservation::custom_header_get", "failed to find " + key);
}


template <typename T>
int dsp::ASCIIObservation::ascii_header_check (const char *header,
    std::string key, const char *format, T result)
{
  int rv = ascii_header_get(header, key.c_str(), format, result);

  if ( rv>0 || !is_required(key) ) 
    return rv;

  throw Error (InvalidState, "ASCIIObservation", "failed load " + key);
}

#endif
