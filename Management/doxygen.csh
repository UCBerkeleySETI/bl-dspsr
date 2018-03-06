#!/bin/csh

doxygen
rm -r web/classes/dspsr
mkdir web/classes
mv doc/html web/classes/dspsr

