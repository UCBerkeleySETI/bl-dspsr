//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DADABuffer_h
#define __DADABuffer_h

#include "dsp/File.h"
#include "dada_hdu.h"

namespace dsp {

  //! Loads BitSeries data using DADA shared memory
  /*! This class pretends to be a file so that it can slip into the
    File::registry */
  class DADABuffer : public File
  {
    
  public:
    
    //! Constructor
    DADABuffer ();
    
    //! Destructor
    ~DADABuffer ();

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename) const;

    //! Seek to the specified time sample
    virtual void seek (int64_t offset, int whence = 0);

    //! Ensure that block_size is an integer multiple of resolution
    virtual void set_block_size (uint64_t _size);

    //! End-of-data is defined by primary read client (passive viewer)
    virtual bool eod();

    //! Get the information about the data source
    virtual void set_info (Observation* obs) { info = obs; }

    //! Reset DADAbuffer
    void rewind ();
 
  protected:

    //! Read the DADA key information from the specified filename
    virtual void open_file (const char* filename);

    //! Close the DADA connection
    void close ();

    //! Load bytes from shared memory
    virtual int64_t load_bytes (unsigned char* buffer, uint64_t bytes);
 
#if HAVE_CUDA
    //! Load bytes from shared memory directory to GPU memory
    int64_t load_bytes_device (unsigned char* device_memory, uint64_t bytes, void * device_handle);
#endif

    //! Set the offset in shared memory
    virtual int64_t seek_bytes (uint64_t bytes);

    //! Over-ride File::set_total_samples
    virtual void set_total_samples ();
   
    //! Shared memory interface
    dada_hdu_t* hdu;

    //! Passive viewing mode
    bool passive;

    //! The byte resolution
    unsigned byte_resolution;

    //! Zero the input data_block after reading values
    char zero_input;

  };

}

#endif // !defined(__DADABuffer_h)
