/*

 */

#ifndef __dsp_UWBFloatUnpacker_h
#define __dsp_UWBFloatUnpacker_h

#include "dsp/Unpacker.h"

namespace dsp {
  
  class UWBFloatUnpacker : public Unpacker
  {
  public:

    //! Constructor
    UWBFloatUnpacker (const char* name = "UWBFloatUnpacker");

    //! Destructor
    ~UWBFloatUnpacker ();

    bool get_order_supported (TimeSeries::Order order) const;
    void set_output_order (TimeSeries::Order order);

    //! Cloner (calls new)
    virtual UWBFloatUnpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);

  protected:
    
    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

  private:

    unsigned ndim;

    unsigned npol;

    bool device_prepared;

    bool first_block;

  };

  class UWBFloatUnpacker::Engine : public Reference::Able
  {
  public:

    virtual void unpack (const BitSeries * input, TimeSeries * output) = 0;

    virtual bool get_device_supported (Memory* memory) const = 0;

    virtual void set_device (Memory* memory) = 0;

    virtual void setup () = 0;

  };

}

#endif
