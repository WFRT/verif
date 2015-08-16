Forecast verification software
==============================

.. image:: https://travis-ci.org/WFRT/verif.svg?branch=master
  :target: https://travis-ci.org/WFRT/verif
.. image:: https://coveralls.io/repos/WFRT/verif/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/WFRT/verif?branch=master

This software computes verification statistics for weather forecasts at point locations. It can be used to
document the quality of one forecasting system but can also be used to compare different weather models and/or
different post-processing methods.

The program works by parsing NetCDF files with observations and forecasts in a specific format (see "Input
files" below).

verif is a command-line tool that can therefore be used to automatically create verification figures.

Developed by Thomas Nipen, David Siuta, and Tim Chui.

.. image:: image.jpg
    :alt: Example plots
    :width: 400
    :align: center

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
**Ubuntu**

Install the required pacakges:

.. code-block:: bash

  sudo apt-get install python-numpy python-scipy python-matplotlib

Then install verif as follows:

.. code-block:: bash

  sudo python setup.py install

This will create the executable /usr/local/bin/verif

**Mac OSX**

Install python, numpy, scipy, and matplotlib, then install verif as follows:

.. code-block:: bash

  sudo python setup.py install

verif will then be installed /usr/local/share/python/ or where ever your python modules are
installed (Look for "Installing verif script to <some directory>" when installing).

Example
-------
.. code-block:: bash

Fake data for testing the program is found in ./examples/. Use the following command to test:

.. code-block:: bash

   verif examples/T_raw.nc examples/T_kf.nc -m mae

Text-based input
----------------
Two data formats are supported. A simple text format for deterministic forecasts has the following format:

.. code-block:: bash

   date     offset id      lat     lon      elev     obs      fcst
   20150101 0      214     49.2    -122.1   92       3.4      2.1
   20150101 1      214     49.2    -122.1   92       4.7      4.2
   20150101 0      180     50.3    -120.3   150      0.2      -1.2

The first line must describe the columns. The following attributes are recognized: date (in
YYYYMMDD), offset (in hours), id (station identifier), lat (in degrees), lon (in degrees),
obs (observations), fcst (deterministic forecast). obs and fcst are required and a value of
0 is used for any missing column. The columns can be in any order. If "id" is not provided, then they
are assigned sequentially starting at 0.

NetCDF input
------------
For more advanced usage, the files  must be in NetCDF and have dimensions and attributes as described below in the
example file. The format is still being decided but will be based on NetCDF/CF standard.

.. code-block:: bash

   netcdf format {
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
      float obs(date, offset, station);              // Observations
      float ens(date, offset, ensemble, station);    // Ensemble forecast
      float fcst(date, offset, station);             // Deterministic forecast
      float cdf(date, offset, threshold, station);   // Accumulated prob at threshold
      float pdf(date, offset, threshold, station);   // Pdf at threshold
      float x(date, offset, quantile, station);      // Threshold corresponding to quantile
      float pit(date, offset, station);              // CDF for threshold=observation

   global attributes:
      : name = "raw";                                // Used as configuration name
      : long_name = "Temperature";                   // Used to label plots
      : standard_name = "air_temperature_2m";
      : Units = "^oC";
      : Conventions = "verif_1.0.0";
      }

Copyright and license
---------------------

Copyright © 2015 UBC Weather Forecast Research Team. verif is licensed under the 3-clause BSD license. See LICENSE
file.
