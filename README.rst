Forecast verification software
==============================

.. image:: https://travis-ci.org/WFRT/verif.svg?branch=master
  :target: https://travis-ci.org/WFRT/verif
.. image:: https://coveralls.io/repos/WFRT/verif/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/WFRT/verif?branch=master

``verif`` is a command-line tool that lets you verify the quality of weather forecasts for point
locations. It can also compare forecasts from different forecasting systems (that have different
models, post-processing methods, etc).

The program reads files with observations and forecasts in a specific format (see "Input files"
below). The input files contain information about dates, forecast lead times, and locations such
that statistics can be aggregated across different dimensions. To ensure a fair comparison among
files, ``verif`` will discard data points where one or more forecast systems have missing forecasts.
Since ``verif`` is a command-line tool, it can be used in scripts to automatically create
verification figures.

A prototype version has been released (see "Installation Instruction" below). We welcome suggestions
for improvements. ``verif`` is developed by Thomas Nipen, David Siuta, and Tim Chui.

.. image:: image.jpg
    :alt: Example plots
    :width: 400
    :align: center

Features
--------

* Deterministic metrics such as MAE, bias, correlation, RMSE (e.g. ``-m mae``)
* Threshold-based metrics such as the false alarm rate, ETS, EDI, Yule's Q (e.g. ``-m ets``)
* Probabilistic metrics such as brier score, PIT-histogram, reliability diagrams (e.g. ``-m bs``)
* Special plots like Taylor diagrams (``-m taylor``), quantile-quantile plots (``-m qq``).
* Plot scores as a function of date, lead time, station elevation/lat/longitude (e.g. ``-x date``)
* Show scores on maps (``-type map``)
* Subset the data by specifying a date range and lat/lon range (``-llrange 5,10,58 60``)
* Export to text (``-type text``)
* Options to adjust font sizes, label positions, tick marks, legends, etc (``-labfs 14``)
* Anomaly statistics relative to a baseline like climatology (``-c climfile.txt``)
* Output to png, jpeg, eps, etc and specify dimensions and resolution (``-f image.png -dpi 300``)

For a full list, run ``verif`` without arguments.

Requirements
------------

* Python
* matplotlib
* numpy
* scipy

Installation Instructions
-------------------------

Download the source code of the prototype version: https://github.com/WFRT/verif/releases/. Unzip
the file and navigate into the extracted folder.

**Ubuntu**

Install the required pacakges:

.. code-block:: bash

  sudo apt-get install python-numpy python-scipy python-matplotlib

Then install ``verif`` by executing the following inside the extracted folder:

.. code-block:: bash

  sudo python setup.py install

This will create the executable ``/usr/local/bin/verif``.  Add this to your PATH environment
variable if necessary (i.e add ``export PATH=/usr/local/bin/:$PATH`` to ``~/.bashrc``). If you do
not have sudo privileges do:

.. code-block:: bash

  sudo python setup.py install --user

This will create the executable ``~/.local/bin/verif``. Add the folder to your PATH environment
variable.

**Mac OSX**

Install python, numpy, scipy, and matplotlib, then install ``verif`` by executing the following
inside the extracted folder:

.. code-block:: bash

  sudo python setup.py install

``verif`` will then be installed ``/usr/local/share/python/`` or where ever your python modules are
installed (Look for "Installing verif script to <some directory>" when installing). Add the folder
to your PATH environment variable.

Examples
--------
Fake data for testing the program is found in ``./examples/``. There is one "raw" forecast file and
one bias-corrected forecast file (where a Kalman filter has been applied). Here are some example
commands to test out:

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
We are working on defining a NetCDF format that can also be read by ``verif``. Here is our current
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
      : Units = "^oC";                               // Used to label axes
      : Conventions = "verif_1.0.0";
      }

Copyright and license
---------------------

Copyright © 2015 UBC Weather Forecast Research Team. ``verif`` is licensed under the 3-clause BSD
license. See LICENSE file.
