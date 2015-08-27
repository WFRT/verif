Forecast verification software
==============================

.. image:: https://travis-ci.org/WFRT/verif.svg?branch=master
  :target: https://travis-ci.org/WFRT/verif
.. image:: https://coveralls.io/repos/WFRT/verif/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/WFRT/verif?branch=master

This program plots verification scores for weather forecasts at point locations. It can be used to
document the quality of one forecasting system but also to compare forecasts from different weather
models and/or where different post-processing methods have been applied.

The program works by reading files with observations and forecasts in a specific format (see "Input
files" below). The files contain information about dates, forecast lead times, and locations such
that statistics can be aggregated across different dimensions.

verif is a command-line tool that can therefore be used to automatically create verification
figures. The statistics can also be output in text format.

verif is developed by Thomas Nipen, David Siuta, and Tim Chui.

.. image:: image.jpg
    :alt: Example plots
    :width: 400
    :align: center

Features
--------

* Deterministic metrics such as MAE, bias, RMSE (e.g. ``-m mae``)
* Threshold-based metrics such as the false alarm rate, ETS, EDI, Yule's Q (e.g. ``-m ets``)
* Probabilistic metrics such as brier score, PIT-histogram, reliability diagrams (e.g. ``-m bs``)
* Special plots like Taylor diagrams (``-m taylor``), error decomposition (``-m error``),
  quantile-quantile plots (``-m qq``).
* Plot scores as a function of date, lead time, station elevation/lat/longitude (e.g. ``-x date``)
* Show scores on maps (``-type map``)
* Subset the data by specifying a date range and lat/lon range (``-llrange 5,10,58 60``)
* Export to text (``-type text``)
* Options to adjust font sizes, label positions, tick marks, legends, etc (``-labfs 14``)
* Anomaly statistics relative to a baseline like climatology (``-c climfile.txt``)
* Output to png, jeg, eps, etc and specify image dimensions and resolution
  (``-f image.png -fs 10,5 -dpi 300``)

For a full list, run verif without arguments.

Requirements
------------

* Python
* matplotlib
* numpy
* scipy

Installation Instructions
-------------------------

Download the source code of a released version: https://github.com/WFRT/verif/releases/

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

Examples
--------
Fake data for testing the program is found in ``./examples/``. There is one "raw" forecast file and
one bias-corrected forecast file (a Kalman filter has been applied). Here are some example commands
to test out:

.. code-block:: bash

   verif examples/raw.txt examples/kf.txt -m mae
   verif examples/raw.txt examples/kf.txt -m ets
   verif examples/raw.txt examples/kf.txt -m taylor
   verif examples/raw.txt examples/kf.txt -m error
   verif examples/raw.txt examples/kf.txt -m reliability -r 0
   verif examples/raw.txt examples/kf.txt -m pithist

Text-based input
----------------
The easiest option is to put the data into the following format:

.. code-block:: bash

   # variable: Temperature
   # units: $^oC$
   date     offset id      lat     lon      elev     obs      fcst   p10
   20150101 0      214     49.2    -122.1   92       3.4      2.1    0.914
   20150101 1      214     49.2    -122.1   92       4.7      4.2    0.858
   20150101 0      180     50.3    -120.3   150      0.2      -1.2   0.992

Any lines starting with '#' can be metadata (currently variable: and units: are recognized). After
that is a header line that must describe the data columns below. The following attributes are
recognized: date (in YYYYMMDD), offset (in hours), id (station identifier), lat (in degrees), lon
(in degrees), obs (observations), fcst (deterministic forecast), p<number> (cumulative probability
at a threshold of 10). obs and fcst are required columns: a value of 0 is used for any missing
column. The columns can be in any order. If 'id' is not provided, then they are assigned
sequentially starting at 0.

Deterministic forecasts will only have "obs" and "fcst", however probabilistic forecasts can provide
any number of cumulative probabilities. For probabilistic forecasts, "fcst" could represent the
ensemble mean (or any other method to reduce the ensemble to a deterministic forecast).

Proposed NetCDF input
---------------------
We are working on defining a NetCDF format that can also be read by verif. Here is our current
proposal, based on the NetCDF/CF standard:

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

Copyright © 2015 UBC Weather Forecast Research Team. verif is licensed under the 3-clause BSD
license. See LICENSE file.
