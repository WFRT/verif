Verification of weather forecasts
=================================

.. image:: https://travis-ci.org/tnipen/verif.svg?branch=master
    :target: https://travis-ci.org/tnipen/verif

This software computes verification statistics for weather forecasts at point locations. It can be used to
document the quality of one forecasting system but can also be used to compare different weather models and/or
different post-processing methods.

The program works by parsing NetCDF files with observations and forecasts in a specific format (see "Input
files" below).

verif is a command-line tool that can therefore be used to automatically create verification figures.

Developed by Thomas Nipen, David Siuta, and Tim Chui.

Features
--------

* Deterministic metrics such as MAE, bias, RMSE
* Threshold-based metrics such as the equitable threat score (ETS)
* Probabilistic metrics such as brier score, PIT-histogram, reliability diagrams
* Plot statistics as a function of date, forecast horizon, station elevation, latitude, or longitude
* Show statistics on maps
* Export to text
* Options to adjust font sizes, label positions, tick marks, legends, etc
* Anomaly statistics (relative to a baseline like climatology)
* Output to png, jeg, eps, etc and specify image dimensions and DPI.

For a full list run verif without arguments.

Requirements
------------

* Python
* matplotlib
* numpy
* scipy

Installation Instructions
-------------------------
To install, just execute:

.. code-block:: bash

  python setup.py install

verif will then be installed /usr/local/share/python/ or where ever your python modules are
installed (Look for "Installing verif script to <some directory>" when installing).Be sure to add this directory
to your $PATH environment variable.

Example
-------
.. code-block:: bash

Fake data for testing the program is found in ./examples/. Use the following command to test:

.. code-block:: bash

   verif examples/T_raw.nc examples/T_kf.nc -m mae

Input files
-----------
Input files must be in NetCDF and have dimensions and attributes as described below in the
example file. The format is still being decided but will be based on NetCDF/CF standard.

.. code-block:: bash

   dimensions :
      date    = UNLIMITED;
      offset  = 48;
      station = 10;
      ensemble = 21;
      threshold = 11;
      quantile = 11;
   variables:
      int id(station);
      int offset(offset);
      int date(date);
      float threshold(threshold);
      float quantile(quantile);
      float lat(station);
      float lon(station);
      float elev(station);
      float obs(date, offset, station);
      float mean(date, offset, station);
      float fcst(date, offset, station);
      float cdf(date, offset, station, threshold);
      float pdf(date, offset, station, threshold);
      float x(date, offset, station, quantile);
      float pit(date, offset, station);

   global attributes:
      : name = "raw";
      : variable = "T";
      : standard_name = "air_temperature_2m";
      : Units = "^oC";
