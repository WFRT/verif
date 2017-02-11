Forecast verification software
==============================

.. image:: https://travis-ci.org/WFRT/verif.svg?branch=master
  :target: https://travis-ci.org/WFRT/verif
.. image:: https://coveralls.io/repos/WFRT/verif/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/WFRT/verif?branch=master

Verif is a command-line tool that lets you verify the quality of weather forecasts for point
locations. It can also compare forecasts from different forecasting systems (that have different
models, post-processing methods, etc).

The program reads files with observations and forecasts in a specific format (see "Input files"
below). The input files contain information about dates, forecast lead times, and locations such
that statistics can be aggregated across different dimensions. To ensure a fair comparison among
files, Verif will discard data points where one or more forecast systems have missing forecasts.
Since Verif is a command-line tool, it can be used in scripts to automatically create
verification figures.

Verif version 1.0 has been released (see "Installation Instruction" below). We welcome suggestions
for improvements. Verif is developed by Thomas Nipen, David Siuta, and Tim Chui.

Features
--------

* Deterministic metrics such as MAE, bias, correlation, RMSE (e.g. ``-m mae``)
* Threshold-based metrics such as the false alarm rate, ETS, EDI, Yule's Q (e.g. ``-m ets``)
* Probabilistic metrics such as brier score, PIT-histogram, reliability diagrams (e.g. ``-m bs``)
* Special plots like Taylor diagrams (``-m taylor``), quantile-quantile plots (``-m qq``).
* Plot scores as a function of date, lead time, station altitude/lat/longitude (e.g. ``-x date``)
* Show scores on maps (``-type map``)
* Subset the data by specifying a date range and lat/lon range (``-latrange 58,60``)
* Export to text (``-type text``)
* Options to adjust font sizes, label positions, tick marks, legends, etc (``-labfs 14``)
* Anomaly statistics relative to a baseline like climatology (``-c climfile.txt``)
* Output to png, jpeg, eps, etc and specify dimensions and resolution (``-f image.png -dpi 300``)

For a full list of all options, run verif on the command-line without arguments, or check the wiki
at https://github.com/WFRT/verif/wiki.

.. image:: image.jpg
    :alt: Example plots
    :width: 400
    :align: center

Installing on Ubuntu
--------------------

**Prerequisites**

Verif requires NetCDF as well as the python packages numpy, scipy, and matplotlib. The python
package mpltoolkits.basemap is optional, but provides a background map when verification scores are
plotted on a map. Install the packages as follows:

.. code-block:: bash

  sudo apt-get update
  sudo apt-get install netcdf-bin libnetcdf-dev libhdf5-serial-dev
  sudo apt-get install python-setuptools python-pip
  sudo apt-get install python-numpy python-scipy python-matplotlib python-mpltoolkits.basemap

**Installing using pip**

After this, the easiest is to install the lastest version of Verif using pip:

.. code-block:: bash

   sudo pip install verif

Verif should then be accessible by typing ``verif`` on the command-line. If you do not have
sudo-rights, then install verif as follows:

.. code-block:: bash

   pip install verif --user

This will create the executable ``~/.local/bin/verif``. Add this to your PATH environment
variable if necessary (i.e add ``export PATH=$PATH:~/.local/bin`` to ``~/.bashrc``).

**Installing from source**

Alternatively, to install from source, download the source code of the latest version:
https://github.com/WFRT/verif/releases/. Unzip the file and navigate into the extracted folder.

Then install Verif by executing the following inside the extracted folder:

.. code-block:: bash

  sudo pip install -r requirements.txt
  sudo python setup.py install

This will create the executable ``/usr/local/bin/verif``. Add ``/usr/local/bin`` to your PATH environment
variable if necessary. If you do not have sudo privileges do:

.. code-block:: bash

  pip install -r requirements.txt --user
  python setup.py install --user

This will create the executable ``~/.local/bin/verif``. Add ``~/.local/bin`` to your PATH environment
variable.

Installing on Mac OSX
---------------------

Follow the proceedure as for Ubuntu (either installing with pip or from source). If installing from
source, then look for the line "Installing verif script to <some directory>", as this will indicate
what folder Verif is installed into. Add the folder to your PATH environment variable if necessary.

Example
--------
A sample dataset for testing the program is found in ``./examples/``. There is one "raw" forecast file and
one "calibrated" forecast file (where statistical methods have been applied). For more information
about the dataset check out the wiki. Here are some example commands to test out:

.. code-block:: bash

   # Shows mean absolute error as a function of lead-time
   verif examples/raw.txt examples/cal.txt -m mae
   # Shows average observed and forecasted values as a function on time
   verif examples/raw.txt examples/cal.txt -m obsfcst -x time
   # Shows equitable threat score as a function of threshold
   verif examples/raw.txt examples/cal.txt -m ets
   # Shows a reliability diagram for a threshold of 0.5 mm
   verif examples/raw.txt examples/cal.txt -m reliability -r 0.5
   # Shows Brier skill score as a function of threshold
   verif examples/raw.txt examples/cal.txt -m bss -x threshold

Text-based input
----------------
To verify your own forecasts, the easiest option is to put the data into the following format:

.. code-block:: bash

   # variable: Temperature
   # units: $^oC$
   date     leadtime location  lat     lon      altitude obs      fcst   p10   q0.1
   20150101 0        214       49.2    -122.1   92       3.4      2.1    0.914 -1.9
   20150101 1        214       49.2    -122.1   92       4.7      4.2    0.858 0.1
   20150101 0        180       50.3    -120.3   150      0.2      -1.2   0.992 -2.1

Any lines starting with '#' can be metadata, currently variable:, units:, x0:, and x1: are
recognized. These are used in labeling axes. x0 can be specified if the variable has a discrete
probability mass at the lower boundary (e.g. 0 for precipitation). Use x1 for the upper boundary
(e.g. 100 % for relative humidity). After that is a header line that must describe the data columns
below. The following attributes are recognized:

* date (in YYYYMMDD)
* unixtime (in seconds since 1970-01-01 00:00:00 +00:00)
* leadtime (forecast lead time in hours)
* location (station identifier)
* lat (in degrees)
* lon (in degrees)
* obs (observations)
* fcst (deterministic forecast)
* p<number> (cumulative probability for a specific threshold, e.g. p10 is the CDF at 10 degrees)
* q<number> (temperature for a specific quantile e.g. q0.1 is the 0.1 quantile)

Either 'date' or 'unixtime' can be supplied. obs and fcst are the only required columns. Note that
the file will likely have many rows with repeated values of leadtime/location/lat/lon/altitude. If
station and lead time information is missing, then Verif assumes they are all for the same
station and lead time. The columns can be in any order.

Deterministic forecasts will only have "obs" and "fcst", however probabilistic forecasts can provide
any number of cumulative probabilities. For probabilistic forecasts, "fcst" could represent the
ensemble mean (or any other method to reduce the ensemble to a deterministic forecast).

For compatibility reason, 'offset' can be used instead of 'leadtime', 'id instead of 'location', and
'elev' instead of 'altitude'.

NetCDF-based  input
---------------------
For larger datasets, the files in NetCDF are much quicker to read. The following dimensions,
variables, and attributes are understood by Verif:

.. code-block:: bash

   netcdf format {
   dimensions:
      time = UNLIMITED;
      leadtime  = 48;
      location = 10;
      ensemble = 21;
      threshold = 11;
      quantile = 11;
   variables:
      int time(time);                                  // Valid time of forecast initialization in
                                                       // number of seconds since 1970-01-01 00:00:00 +00:00
      float leadtime(leadtime);                        // Number of hours since forecast init
      int location(location);                          // Id for each station location
      float threshold(threshold);
      float quantile(quantile);                        // Numbers between 0 and 1
      float lat(location);                             // Decimal degrees latitude
      float lon(location);                             // Decimal degrees longitude
      float altitude(location);                        // Altitude in meters
      float obs(time, leadtime, location);             // Observations
      float fcst(time, leadtime, location);            // Deterministic forecast
      float cdf(time, leadtime, location, threshold);  // Accumulated prob at threshold
      float pdf(time, leadtime, location, threshold);  // Probability density at threshold
      float x(time, leadtime, location, quantile);     // Threshold corresponding to quantile
      float pit(time, leadtime, location);             // CDF for threshold=observation

   // global attributes:
      : long_name = "Precipitation";                   // Used to label axes in plots
      : standard_name = "precipitation_amount";        // NetCDF/CF standard name of the forecast variable
      : x0 = 0;                                        // Discrete mass at lower boundary (e.g. 0 mm for precipitation). Omit otherwise.
      : x1 = 100;                                      // Discrete mass at upper boundary (e.g. 100% for relative humidity). Omit otherwise.
      : verif_version = "1.0.0";                       // Not required, but will be parsed in the future if format changes
      }

Copyright and license
---------------------

Copyright © 2015-2017 UBC Weather Forecast Research Team. Verif is licensed under the 3-clause
BSD license. See LICENSE file.
