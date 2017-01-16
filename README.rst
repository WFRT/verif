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

A beta release of Verif version 1.0 has been released (see "Installation Instruction" below). We
welcome suggestions for improvements. Verif is developed by Thomas Nipen, David Siuta, and Tim Chui.

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
* Plot scores as a function of date, lead time, station altitude/lat/longitude (e.g. ``-x date``)
* Show scores on maps (``-type map``)
* Subset the data by specifying a date range and lat/lon range (``-latrange 58,60``)
* Export to text (``-type text``)
* Options to adjust font sizes, label positions, tick marks, legends, etc (``-labfs 14``)
* Anomaly statistics relative to a baseline like climatology (``-c climfile.txt``)
* Output to png, jpeg, eps, etc and specify dimensions and resolution (``-f image.png -dpi 300``)

For a full list, run Verif without arguments.

Installing on Ubuntu
--------------------

**Prerequisites**

Verif requires NetCDF as well as the python packages numpy, scipy, and matplotlib. The python
package mpltoolkits.basemap is optional, but provides a background map when verification scores are
plotted on a map. Install the packages as follows:

.. code-block:: bash

  sudo apt-get install netcdf-bin libnetcdf-dev libhdf5-serial-dev
  sudo apt-get install python-setuptools python-numpy python-scipy python-matplotlib python-mpltoolkits.basemap

**Installing using pip**
The easiest is to install the lastest version of Verif using pip:

.. code-block:: bash

   sudo pip install verif

Verif should then be accessible type typing ``verif`` on the command-line.

**Installing from source**
Alternatively, to install from source, download the source code of the latest version:
https://github.com/WFRT/verif/releases/. Unzip the file and navigate into the extracted folder.

Then install Verif by executing the following inside the extracted folder:

.. code-block:: bash

  sudo python setup.py install

This will create the executable ``/usr/local/bin/verif``. Add this to your PATH environment
variable if necessary (i.e add ``export PATH=/usr/local/bin/:$PATH`` to ``~/.bashrc``). If you do
not have sudo privileges do:

.. code-block:: bash

  sudo python setup.py install --user

This will create the executable ``~/.local/bin/verif``. Add the folder to your PATH environment
variable.

Installing on Mac OSX
---------------------

Install NetCDF, numpy, scipy, and matplotlib, and basemap (optionally). Then install Verif by
executing the following inside the extracted folder:

.. code-block:: bash

  sudo python setup.py install

Verif will then be installed into ``/usr/local/share/python/`` or where ever your python modules are
installed (Look for "Installing verif script to <some directory>" when installing). Add the folder
to your PATH environment variable, if it is not already added.

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

Available metrics
-----------------
Here is a list of currently supported metrics. Note that the plots that are possible to make depend
on what variables are available in the input files.

======================  ===============================================================
**Deterministic**       **Description**
----------------------  ---------------------------------------------------------------
``-m alphaindex``       Alpha index
``-m bias``             Mean error
``-m cmae``             Cube-root mean absolute cubic error
``-m corr``             Pearson correlation between obs and forecast
``-m derror``           Error in distribution of deterministic values
``-m dmb``              Degree of mass balance (mean obs / mean fcst)
``-m ef``               Exceedance fraction: fraction that fcst > obs
``-m fcst``             Average forecast value
``-m kendallcorr``      Kendall correlation
``-m leps``             Linear error in probability space
``-m mae``              Mean of forecasts
``-m mbias``            Multiplicative bias
``-m nsec``             Nash-Sutcliffe efficiency coefficient
``-m obs``              Mean of observations
``-m rankcorr``         Spearman rank correlation
``-m rmse``             Root mean squared error
``-m rmsf``             Root mean squared factor
``-m stderror``         Standard error
``-m within``           Percentage of forecasts that are within some error bound
----------------------  ---------------------------------------------------------------
**Threshold**           **Description**
----------------------  ---------------------------------------------------------------
``-m a``                Fraction of events that are hits
``-m b``                Fraction of events that are false alarms
``-m baserate``         Climatological frequency
``-m biasfreq``         Numer of forecasts / number of observations
``-m c``                Fraction of events that are misses
``-m d``                Fraction of events that are correct rejections
``-m diff``             Difference between false alarms and misses
``-m dscore``           Generalized discrimination score
``-m edi``              Extremal dependency index
``-m eds``              Extreme dependency score
``-m ets``              Equitable threat score
``-m fa``               False alarm rate
``-m far``              False alarm ratio
``-m fcstrate``         Fractions of forecasts (a + b)
``-m hit``              Hit rate
``-m hss``              Heidke skill score
``-m kss``              Hanssen-Kuiper skill score
``-m lor``              Log odds ratio
``-m miss``             Miss rate
``-m n``                Total cases (a + b + c + d)
``-m or``               Odds ratio
``-m pc``               Proportions correct
``-m sedi``             Symmetric extremal dependency index
``-m seds``             Symmetric extreme dependency score
``-m threat``           Threat score
``-m yulesq``           Yule's Q (odds ratio skill score)
----------------------  ---------------------------------------------------------------
**Probabilistic**       **Description**
----------------------  ---------------------------------------------------------------
``-m bs``               Brier score
``-m bsrel``            Reliability component of Brier score
``-m bsres``            Resolution component of Brier score
``-m bss``              Brier skill score
``-m bsunc``            Uncertainty component of Brier score
``-m ign0``             Ignorance of the binary probability based on threshold
``-m marginalratio``    Ratio of marginal probability of obs to that of fcst
``-m pitdev``           Deviation of the PIT histogram
``-m quantilescore``    Quantile score
``-m spherical``        Pherical probabilistic scoring rule
----------------------  ---------------------------------------------------------------
**Special plots**       **Description**
----------------------  ---------------------------------------------------------------
``-m against``          Plots the determinstic forecasts from each file against each other
``-m change``           Forecast skill (MAE) as a function of change in obs from previous forecast run
``-m cond``             Plots forecasts as a function of obs
``-m discrimination``   Discrimination diagram for a specified threshold
``-m droc``             Receiver operating characteristic for deterministic forecast
``-m droc0``            Like droc, except don't use different forecast thresholds
``-m drocnorm``         Like droc, except trainsform axes using standard normal distribution
``-m economicvalue``    Economic value for a specified threshold
``-m error``            Decomposition of RMSE into systematic and unsystematic components
``-m freq``             Show frequency distribution of obs and fcst
``-m igncontrib``       Shows how much each probability issued contributes to total ignorance
``-m impact``           Compares two forecast inputs and shows where the improvements come from
``-m invreliability``   Reliability diagram for a specified quantile
``-m marginal``         Marginal distribution for a specified threshold
``-m meteo``            Show forecasts and obs in a meteogram
``-m obsfcst``          A plot showing both obs and fcst
``-m performance``      Diagram showing POD, FAR, bias, and threat score
``-m pithist``          Histogram of PIT values
``-m qq``               Quantile-quantile plot
``-m reliability``      Reliability diagram for a specified threshold
``-m roc``              Receiver operating characteristics plot for a specified threshold
``-m scatter``          A scatter plt of obs and fcst
``-m spreadskill``      Plots forecast spread vs forecast skilL
``-m taylor``           Taylor diagram showing correlation and fcst stdev
``-m timeseries``       Time series of obs and forecasts
======================  ===============================================================

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

Any lines starting with '#' can be metadata (currently variable: and units: are recognized). After
that is a header line that must describe the data columns below. The following attributes are
recognized:

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
      : long_name = "Temperature";                     // Used to label axes in plots
      : standard_name = "air_temperature";             // NetCDF/CF standard name of the forecast variable
      : verif_version = "1.0.0";                       // Not required, but will be parsed in the future if format changes
      }

Copyright and license
---------------------

Copyright © 2015-2017 UBC Weather Forecast Research Team. Verif is licensed under the 3-clause
BSD license. See LICENSE file.
