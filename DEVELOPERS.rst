Notes to developers
===================

Metrics and Outputs
-------------------
There are two ways to create a new metric that are recognized by verif's -m option. This is done by
creating new subclasses in Metric.py or Output.py.

1) **Metric**: This class represents any metric that can be computed from a set of forecasts
   and observation pairs, regardless of how this set has been put together. This allows the user to
   choose how to stratify the results, i.e. as a function of leadtime, date, location, etc. Most
   scores can be implemented in this framework.

   The **Deterministic** class represents any class that can compute its score based only on pairs
   of observations and the deterministic forecast (i.e. not using any forecast probabilities).

   The **Contingency** class is useful for any score that uses the 4 values in the 2x2 contingency
   table created when using an exceedance threshold for both the observations and the deterministic
   forecast.

   The class has a number of attributes that should be filled in by any new metrics. These are
   documented in the code and include attributes such as what the range of the metric is, and what
   value of the score indicates a perfect score.

   New subclass added to Metric.py will automatically be accesible by verif's -m option. Use the
   class name converted to lower case.

2) **Output**: This class represents any plot that can be generated based on the input data. The
   **Default** output will plot scores for any Metric as a function of leadtime, date, etc. However,
   specialized plots can be made such as the reliability diagram. This plot does not fit into the
   standard framework for a Metric and has therefore been implemented in this more general class.

   The Output class allows much more flexible plots to be made, than by subclassing Metric. However
   an Output cannot make use of the functionality that verif provides in terms of aggregating
   statistics across different leadtimes, dates, etc. Aggregation must therefore either be
   implemented again, or not be supported by the Output.

   Unlike a new Metric subclass, an Output subclass must be registered in Driver.py before it will
   be accessible by verif's -m option.


Input data
----------
An **Input** represents observations and forecasts for one forecast variable (such as temperature)
for one forecasting system. The input has obs/fcsts for different leadtimes, dates, and locations.

To allow a fair comparison of mulitple inputs, only leadtimes, dates, and locations for which all
inputs have available forecasts should be used in the calculations. To allow for this, the inputs
are collected into a **Data** class. This is the main interface used by the rest of verif to access
the data in the raw input files.


Releasing a new version
-----------------------
1) Commit your changes.
2) Determine the new tag. Use "git tag" to find the latest tag and increment according to
   http://semver.org/. The tag should be in the form v0.1.1.
3) Edit verif/Version.py to indicate the new version (without the v-prefix). This value is used
   by verif --version.
4) Update the debian changelog by putting in the version number and filling in a description
   of the new release.

  .. code-block:: bash

     dch -i

5) Run the test suite and make sure there are no failures.

  .. code-block:: bash

    nosetests

6) Commit the release information

  .. code-block:: bash

    git commit debian/changelog verif/Version.py

7) Tag the version in git (using the previously determined tag)

  .. code-block:: bash

     git tag <tag including the v-prefix>

8) Push the release to the repository

  .. code-block:: bash

     git push --tags origin master
