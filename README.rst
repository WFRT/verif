Forecast verification software
==============================
.. image:: https://img.shields.io/github/v/release/WFRT/verif.svg
  :target: https://github.com/WFRT/verif/releases
  :alt: Release badge
.. image:: https://github.com/WFRT/verif/actions/workflows/python-package.yml/badge.svg
  :target: https://github.com/WFRT/verif/actions/workflows/python-package.yml
  :alt: CI badge
.. image:: https://img.shields.io/badge/DOI-10.1175%2FBAMS--D--22--0253.1-blue
  :target: https://doi.org/10.1175/BAMS-D-22-0253.1
  :alt: Bams Article

Verif is a command-line tool that lets you verify the quality of weather forecasts for point
locations. It can also compare forecasts from different forecasting systems (that have different
models, post-processing methods, etc).

The program reads files with observations and forecasts in a specific format (see "Input files"
below). The input files contain information about dates, forecast lead times, and locations such
that statistics can be aggregated across different dimensions. To ensure a fair comparison among
files, Verif will discard data points where one or more forecast systems have missing forecasts.
Since Verif is a command-line tool, it can be used in scripts to automatically create
verification figures.

Verif version 1.3 has been released (see "Installation Instruction" below). We welcome suggestions
for improvements. Verif is developed by Thomas Nipen (thomasn@met.no), with contributions from many.

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

Resources
---------

* Check out the **wiki** at https://github.com/WFRT/verif/wiki.
* Found a bug? Please report it in the **issue tracker** at https://github.com/WFRT/verif/issues.
* Reach out to the Verif community in the **discussions** at https://github.com/WFRT/verif/discussions.
* Check out our `article <https://journals.ametsoc.org/view/journals/bams/104/9/BAMS-D-22-0253.1.xml>`_ published in BAMS.

.. image:: other/image.jpg
    :alt: Example plots
    :width: 400
    :align: center

Getting started
---------------

The easiest way to get started is to install verif with pip:

.. code-block:: bash

    pip3 install verif

The `verif` command-line program should then be available.

Note that the python package "cartopy" is optional and not installed by default when verif is
installed, Verif uses "cartopy" for creating a background map when verification scores are plotted
on a map. To install Cartopy run

.. code-block:: bash

    pip3 install cartopy

To upgrade to a newer version of Verif, run the following:

.. code-block:: bash

   pip3 install verif --upgrade

Examples
--------
To test Verif, you can download example datasets from the github
`discussion page <https://github.com/WFRT/verif/discussions>`_. For example, download the following two files from the wind speed dataset: `MEPS.nc <https://thredds.met.no/thredds/fileServer/metusers/thomasn/verif_datasets/short_range_wind/MEPS.nc>`_ (2.5 km regional model; 20MB file size) and `ECMWF.nc <https://thredds.met.no/thredds/fileServer/metusers/thomasn/verif_datasets/short_range_wind/ECMWF.nc>`_ (0.2° global model; 24MB file size). Then run the following
commands to test out the software:

.. code-block:: bash

   # Shows mean absolute error as a function of lead-time
   verif MEPS.nc ECMWF.nc -m mae
   # Shows average observed and forecasted values as a function on time
   verif MEPS.nc ECMWF.nc -m obsfcst -x time
   # Shows equitable threat score as a function of threshold
   verif MEPS.nc ECMWF.nc -m ets
   # Shows a reliability diagram for a threshold of 13.9 m/s (gale force winds)
   verif MEPS.nc ECMWF.nc -m reliability -r 13.9
   # Shows Brier skill score as a function of threshold
   verif MEPS.nc ECMWF.nc -m bss -x threshold

How to cite
-----------
Nipen, T. N., R. B. Stull, C. Lussana, and I. A. Seierstad, 2023: `Verif: A Weather-Prediction Verification Tool for Effective Product Development <https://journals.ametsoc.org/view/journals/bams/104/9/BAMS-D-22-0253.1.xml>`_. Bull. Amer. Meteor.  Soc., 104, E1610–E1618, https://doi.org/10.1175/BAMS-D-22-0253.1.

Copyright and license
---------------------

Copyright © 2013-2024 UBC Weather Forecast Research Team. Verif is licensed under the 3-clause
BSD license. See LICENSE file.
