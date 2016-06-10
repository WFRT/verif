import numpy as np
import verif.Util as Util
import sys
import inspect


# Returns a list of all metric classes
def getAllMetrics():
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return temp


# Returns a metric object of a class with the given name
def getMetric(name):
   metrics = getAllMetrics()
   m = None
   for mm in metrics:
      if(name == mm[0].lower() and mm[1].isValid()):
         m = mm[1]()
   return m


# Computes scores for each xaxis value
class Metric:
   # Overload these variables
   _min = None  # Minimum value this metric can produce
   _max = None  # Maximum value this mertic can produce
   _defaultAxis = "offset"  # If no axis is specified, use this axis as default
   _defaultBinType = None
   _reqThreshold = False  # Does this metric require thresholds?
   _supThreshold = False  # Does this metric support thresholds?
   _experimental = False  # Is this metric not fully tested yet?
   _perfectScore = None
   _aggregator = np.mean
   _aggregatorName = "mean"
   _supAggregator = False  # Does this metric use self._aggregator?
   _orientation = 0  # 1 for +, -1 for -, 0 for all other
   # Information about metric. The y-axis label is controlled by self.label()
   # Also, self.name() is the name of the metric

   # A short one-liner describing the metric. This will show up in the
   # main verif documentation.
   _description = ""
   # A longer description. This will show up in the documentation when a
   # specific metric is chosen.
   _long = None
   _reference = None  # A string with an academic reference

   # Compute the score
   # data: use getScores([metric1, metric2...]) to get data
   #       data has already been configured to only retrieve data along a
   #       certain dimension
   # tRange: [lowerThreshold, upperThreshold]
   def compute(self, data, tRange):
      # assert(isinstance(tRange, list))
      # assert(len(tRange) == 2)
      size = data.getAxisSize()
      scores = np.zeros(size, 'float')
      # Loop over x-axis
      for i in range(0, size):
         data.setIndex(i)
         x = self.computeCore(data, tRange)
         scores[i] = x
      return scores

   # returns 1 for a positively oriented score (higher values are better),
   # -1 for negative, and 0 for all others
   @classmethod
   def orientation(cls):
      return cls._orientation

   # Implement this
   def computeCore(self, data, tRange):
      Util.error("Metric '" + self.getClassName() +
            "' has not been implemented yet")

   @classmethod
   def description(cls):
      return cls._description

   @classmethod
   def reference(cls):
      return cls._reference

   # Is this a valid metric that should be created be called?
   @classmethod
   def isValid(cls):
      return cls.summary() is not ""

   @classmethod
   def help(cls):
      s = cls.description()
      if(cls.orientation is not 0):
         s = s + "\n" + Util.green("Orientation: ")
         if(cls.orientation == 1):
            s = s + "Positive"
         elif(cls.orientation == -1):
            s = s + "Negative"
         else:
            s = s + "None"
      if(cls.perfectScore is not None):
         s = s + "\n" + Util.green("Perfect score: ") + str(cls._perfectScore)
      if(cls.min is not None):
         s = s + "\n" + Util.green("Minimum value: ") + str(cls._min)
      if(cls.max is not None):
         s = s + "\n" + Util.green("Maximum value: ") + str(cls._max)
      if(cls._long is not None):
         s = s + "\n" + Util.green("Description: ") + cls._long
      if(cls.reference() is not None):
         s = s + "\n" + Util.green("Reference: ") + cls.reference()
      return s

   @classmethod
   def summary(cls):
      desc = cls.description()
      if(desc == ""):
         return ""
      extra = ""
      # if(cls._experimental):
      #    extra = " " + Util.experimental() + "."
      if(cls._supAggregator):
         extra = " Supports -ct."
      if(cls._perfectScore is not None):
         extra = extra + " " + "Perfect score " + str(cls._perfectScore) + "."
      return desc + "." + extra

   # Does this metric require thresholds in order to be computable?
   @classmethod
   def requiresThresholds(cls):
      return cls._reqThreshold

   # If this metric is to be plotted, along which axis should it be plotted by
   # default?
   @classmethod
   def defaultAxis(cls):
      return cls._defaultAxis

   @classmethod
   def defaultBinType(cls):
      return cls._defaultBinType

   @classmethod
   def perfectScore(cls):
      return cls._perfectScore

   # Does it make sense to use '-x threshold' with this metric?
   @classmethod
   def supportsThreshold(cls):
      return cls._supThreshold

   # Minimum value the metric can take on
   @classmethod
   def min(cls):
      return cls._min

   # Maximum value the metric can take on
   @classmethod
   def max(cls):
      return cls._max

   def setAggregator(self, name):
      self._aggregatorName = name
      if(name == "mean"):
         self._aggregator = np.mean
      elif(name == "median"):
         self._aggregator = np.median
      elif(name == "min"):
         self._aggregator = np.min
      elif(name == "max"):
         self._aggregator = np.max
      elif(name == "std"):
         self._aggregator = np.std
      elif(name == "range"):
         self._aggregator = Util.nprange
      elif(name == "count"):
         self._aggregator = Util.numvalid
      elif(name == "meanabs"):
         self._aggregator = Util.meanabs
      elif(Util.isnumeric(name)):
         quantile = float(name)
         if quantile < 0 or quantile > 1:
            Util.error("Number after -ct must must be between 0 and 1")

         def func(x):
            return np.percentile(x, quantile*100)
         self._aggregator = func
      else:
         Util.error("Invalid aggregator")

   @classmethod
   def getClassName(cls):
      name = cls.__name__
      return name

   # This is the y-axis label for this metric. Override this if
   # the metric does not have the same units as the forecast variable
   def label(self, data):
      return self.name() + " (" + data.getUnits() + ")"

   # Cannot be a classmethod, since it might use self._aggregator
   def name(self):
      return self.getClassName()

   def label(self, data):
      return self.name() + " (" + data.getUnits() + ")"

   # Is x within the range?
   @staticmethod
   def within(x, range):
      # TODO: Which is correct?
      # The second is best for precip, when doing Brier score -r 0
      # return (x >= range[0]) & (x < range[1])
      return (x > range[0]) & (x <= range[1])


class Default(Metric):
   # aux: When reading the score, also pull values for 'aux' to ensure
   # only common data points are returned
   def __init__(self, name, aux=None):
      self._name = name
      self._aux = aux

   def computeCore(self, data, tRange):
      if(self._aux is not None):
         [values, aux] = data.getScores([self._name, self._aux])
      else:
         values = data.getScores(self._name)
      return self._aggregator(values)

   def name(self):
      return self._aggregatorName.title() + " of " + self._name


class Mean(Metric):
   def __init__(self, metric):
      self._metric = metric

   def computeCore(self, data, tRange):
      return np.mean(self._metric.compute(data, tRange))

   def name(self):
      return "Mean of " + self._metric.name()


# Note: This cannot be a subclass of Deterministic, since we don't want
# to remove obs for which the forecasts are missing. Same for Fcst.
class Obs(Metric):
   _description = "Observed value"
   _supAggregator = True
   _orientation = 0

   def computeCore(self, data, tRange):
      obs = data.getScores("obs")[0]
      return self._aggregator(obs)

   def name(self):
      return self._aggregatorName.title() + " of observation"


class Fcst(Metric):
   _description = "Forecasted value"
   _supAggregator = True
   _orientation = 0

   def computeCore(self, data, tRange):
      fcst = data.getScores("fcst")[0]
      return self._aggregator(fcst)

   def name(self):
      return self._aggregatorName.title() + " of forecast"


# Deterministic metric which is dependent only on "obs" and "fcst"
class Deterministic(Metric):
   def computeCore(self, data, tRange):
      [obs, fcst] = data.getScores(["obs", "fcst"])
      return self.computeObsFcst(obs, fcst)

   def computeObsFcst(self, obs, fcst):
      assert(obs.shape[0] == fcst.shape[0])
      # Remove missing values
      I = np.where((np.isnan(obs) | np.isnan(fcst)) == 0)[0]
      obs = obs[I]
      fcst = fcst[I]
      if(obs.shape[0] > 0):
         return self._computeObsFcst(obs, fcst)
      else:
         return np.nan

   # Subclass must implement this function:
   # Preconditions for obs and fcst:
   #     - obs and fcst are the same length
   #     - length >= 1
   #     - no missing values
   def _computeObsFcst(self, obs, fcst):
      Util.error("Metric " + self.name() +
            " has not implemented _computeObsFcst()")


class Mae(Deterministic):
   _min = 0
   _description = "Mean absolute error"
   _perfectScore = 0
   _supAggregator = True

   def _computeObsFcst(self, obs, fcst):
      return self._aggregator(abs(obs - fcst))

   def name(self):
      return "MAE"


class Bias(Deterministic):
   _description = "Bias"
   _perfectScore = 0
   _supAggregator = True
   _orientation = 0

   def _computeObsFcst(self, obs, fcst):
      return self._aggregator(obs - fcst)


class Ef(Deterministic):
   _description = "Exeedance fraction: percentage of times that forecasts"\
                  " > observations"
   _min = 0
   _max = 100
   _perfectScore = 50
   _orientation = 0

   def _computeObsFcst(self, obs, fcst):
      Nfcst = np.sum(obs < fcst)
      return Nfcst / 1.0 / len(fcst) * 100

   def name(self):
      return "Exceedance fraction"

   def label(self, data):
      return "% times fcst > obs"


class StdError(Deterministic):
   _min = 0
   _description = "Standard error (i.e. RMSE if forecast had no bias)"
   _perfectScore = 0
   _orientation = -1

   def _computeObsFcst(self, obs, fcst):
      bias = np.mean(obs - fcst)
      return np.mean((obs - fcst - bias) ** 2) ** 0.5

   def name(self):
      return "Standard error"


class Rmse(Deterministic):
   _min = 0
   _description = "Root mean squared error"
   _perfectScore = 0
   _orientation = -1

   def _computeObsFcst(self, obs, fcst):
      return np.mean((obs - fcst) ** 2) ** 0.5

   def name(self):
      return "RMSE"


class Rmsf(Deterministic):
   _min = 0
   _description = "Root mean squared factor"
   _perfectScore = 1
   _orientation = 0

   def _computeObsFcst(self, obs, fcst):
      return np.exp(np.mean((np.log(fcst / obs)) ** 2) ** 0.5)

   def name(self):
      return "RMSF"


class Crmse(Deterministic):
   _min = 0
   _description = "Centered root mean squared error (RMSE without bias)"
   _perfectScore = 0
   _orientation = -1

   def _computeObsFcst(self, obs, fcst):
      bias = np.mean(obs) - np.mean(fcst)
      return np.mean((obs - fcst - bias) ** 2) ** 0.5

   def name(self):
      return "CRMSE"


class Cmae(Deterministic):
   _min = 0
   _description = "Cube-root mean absolute cubic error"
   _perfectScore = 0
   _orientation = -1

   def _computeObsFcst(self, obs, fcst):
      return (np.mean(abs(obs ** 3 - fcst ** 3))) ** (1.0 / 3)

   def name(self):
      return "CMAE"


class Nsec(Deterministic):
   _min = 0
   _description = "Nash-Sutcliffe efficiency coefficient"
   _perfectScore = 1
   _orientation = 1
   _max = 1

   def _computeObsFcst(self, obs, fcst):
      meanobs = np.mean(obs)
      num = np.sum((fcst - obs) ** 2)
      denom = np.sum((obs - meanobs) ** 2)
      if(denom == 0):
         return np.nan
      else:
         return 1 - num / denom

   def name(self):
      return "NSEC"

   def label(self, data):
      return self.name()


class Alphaindex(Deterministic):
   _min = 0
   _description = "Alpha index"
   _perfectScore = 0
   _orientation = -1
   _max = 2
   _min = 0

   def _computeObsFcst(self, obs, fcst):
      meanobs = np.mean(obs)
      meanfcst = np.mean(fcst)
      num = np.sum((fcst - obs - meanfcst + meanobs) ** 2)
      denom = np.sum((fcst - meanfcst) ** 2 + (obs - meanobs) ** 2)
      if(denom == 0):
         return np.nan
      else:
         return 1 - num / denom

   def name(self):
      return "Alpha index"

   def label(self, data):
      return self.name()


class Leps(Deterministic):
   _min = 0
   _description = "Linear error in probability space"
   _perfectScore = 0
   _orientation = -1

   def _computeObsFcst(self, obs, fcst):
      N = len(obs)
      # Compute obs quantiles
      Iobs = np.array(np.argsort(obs), 'float')
      qobs = Iobs / N

      # Compute the quantiles that the forecasts are relative
      # to the observations
      qfcst = np.zeros(N, 'float')
      sortobs = np.sort(obs)
      for i in range(0, N):
         I = np.where(fcst[i] < sortobs)[0]
         if(len(I) > 0):
            qfcst[i] = float(I[0]) / N
         else:
            qfcst[i] = 1
      return np.mean(abs(qfcst - qobs))

   def name(self):
      return "LEPS"


class Dmb(Deterministic):
   _description = "Degree of mass balance (obs/fcst)"
   _perfectScore = 1
   _orientation = 0

   def _computeObsFcst(self, obs, fcst):
      return np.mean(obs) / np.mean(fcst)

   def name(self):
      return "Degree of mass balance (obs/fcst)"


class Mbias(Deterministic):
   _description = "Multiplicative bias (obs/fcst)"
   _perfectScore = 1
   _orientation = 0

   def _computeObsFcst(self, obs, fcst):
      return (np.mean(obs) / np.mean(fcst))

   def name(self):
      return self._description

   def label(self, data):
      return self._description


class Corr(Deterministic):
   _min = 0  # Technically -1, but values below 0 are not as interesting
   _max = 1
   _description = "Correlation between obesrvations and forecasts"
   _perfectScore = 1
   _orientation = 1

   def _computeObsFcst(self, obs, fcst):
      if(len(obs) <= 1):
         return np.nan
      if(np.var(fcst) == 0):
         return np.nan
      return np.corrcoef(obs, fcst)[1, 0]

   def name(self):
      return "Correlation"

   def label(self, data):
      return "Correlation"


class RankCorr(Deterministic):
   _min = 0  # Technically -1, but values below 0 are not as interesting
   _max = 1
   _description = "Rank correlation between obesrvations and forecasts"
   _perfectScore = 1
   _orientation = 1

   def _computeObsFcst(self, obs, fcst):
      import scipy.stats
      if(len(obs) <= 1):
         return np.nan
      return scipy.stats.spearmanr(obs, fcst)[0]

   def name(self):
      return "Rank correlation"

   def label(self, data):
      return "Rank correlation"


class KendallCorr(Deterministic):
   _min = 0  # Technically -1, but values below 0 are not as interesting
   _max = 1
   _description = "Kendall correlation between obesrvations and forecasts"
   _perfectScore = 1
   _orientation = 1

   def _computeObsFcst(self, obs, fcst):
      import scipy.stats
      if(len(obs) <= 1):
         return np.nan
      if(np.var(fcst) == 0):
         return np.nan
      return scipy.stats.kendalltau(obs, fcst)[0]

   def name(self):
      return "Kendall correlation"

   def label(self, data):
      return "Kendall correlation"


class Extreme(Metric):
   def calc(self, data, func, variable):
      [value] = data.getScores([variable])
      if(len(value) == 0):
         return np.nan
      return func(value)


class MaxObs(Extreme):
   _description = "Maximum observed value"
   _orientation = 0

   def computeCore(self, data, tRange):
      return self.calc(data, np.max, "obs")


class MinObs(Extreme):
   _description = "Minimum observed value"
   _orientation = 0

   def computeCore(self, data, tRange):
      return self.calc(data, np.min, "obs")


class MaxFcst(Extreme):
   _description = "Maximum forecasted value"
   _orientation = 0

   def computeCore(self, data, tRange):
      return self.calc(data, np.max, "fcst")


class MinFcst(Extreme):
   _description = "Minimum forecasted value"
   _orientation = 0

   def computeCore(self, data, tRange):
      return self.calc(data, np.min, "fcst")


# Returns all PIT values
class Pit(Metric):
   _min = 0
   _max = 1
   _orientation = 0

   def __init__(self, name="pit"):
      self._name = name

   def label(self, data):
      return "PIT"

   def compute(self, data, tRange):
      x0 = data.getVariable().getX0()
      x1 = data.getVariable().getX1()
      if(x0 is None and x1 is None):
         [pit] = data.getScores([self._name])
      else:
         [obs, pit] = data.getScores(["obs", self._name])
         if(x0 is not None):
            I = np.where(obs == x0)[0]
            pit[I] = np.random.rand(len(I)) * pit[I]
         if(x1 is not None):
            I = np.where(obs == x1)[0]
            pit[I] = 1 - np.random.rand(len(I)) * (1 - pit[I])
         # I = np.where((fcst > 2) & (fcst < 2000))[0]
         # I = np.where((fcst > 20))[0]
         # pit = pit[I]
      return pit

   def name(self):
      return "PIT"


# Returns all PIT values
class PitDev(Metric):
   _min = 0
   # _max = 1
   _perfectScore = 1
   _description = "Deviation of the PIT histogram"
   _orientation = -1

   def __init__(self, numBins=11):
      self._metric = Pit()
      self._bins = np.linspace(0, 1, numBins)

   def label(self, data):
      return "PIT histogram deviation"

   def computeCore(self, data, tRange):
      pit = self._metric.compute(data, tRange)
      pit = pit[np.isnan(pit) == 0]

      nb = len(self._bins) - 1
      D = self.deviation(pit, nb)
      D0 = self.expectedDeviation(pit, nb)
      dev = D / D0
      return dev

   def name(self):
      return "PIT deviation factor"

   @staticmethod
   def expectedDeviation(values, numBins):
      if(len(values) == 0 or numBins == 0):
         return np.nan
      return np.sqrt((1.0 - 1.0 / numBins) / (len(values) * numBins))

   @staticmethod
   def deviation(values, numBins):
      if(len(values) == 0 or numBins == 0):
         return np.nan
      x = np.linspace(0, 1, numBins + 1)
      n = np.histogram(values, x)[0]
      n = n * 1.0 / sum(n)
      return np.sqrt(1.0 / numBins * np.sum((n - 1.0 / numBins) ** 2))

   @staticmethod
   def deviationStd(values, numBins):
      if(len(values) == 0 or numBins == 0):
         return np.nan
      n = len(values)
      p = 1.0 / numBins
      numPerBinStd = np.sqrt(n * p * (1 - p))
      std = numPerBinStd / n
      return std

   # What reduction in ignorance is possible by calibrating the PIT-histogram?
   @staticmethod
   def ignorancePotential(values, numBins):
      if(len(values) == 0 or numBins == 0):
         return np.nan
      x = np.linspace(0, 1, numBins + 1)
      n = np.histogram(values, x)[0]
      n = n * 1.0 / sum(n)
      expected = 1.0 / numBins
      ign = np.sum(n * np.log2(n / expected)) / sum(n)
      return ign


class MarginalRatio(Metric):
   _min = 0
   _description = "Ratio of marginal probability of obs to marginal" \
         " probability of fcst. Use -r."
   _perfectScore = 1
   _reqThreshold = True
   _supThreshold = True
   _defaultAxis = "threshold"
   _experimental = True
   _orientation = 0

   def computeCore(self, data, tRange):
      if(np.isinf(tRange[0])):
         pvar = data.getPvar(tRange[1])
         [obs, p1] = data.getScores(["obs", pvar])
         p0 = 0 * p1
      elif(np.isinf(tRange[1])):
         pvar = data.getPvar(tRange[0])
         [obs, p0] = data.getScores(["obs", pvar])
         p1 = 0 * p0 + 1
      else:
         pvar0 = data.getPvar(tRange[0])
         pvar1 = data.getPvar(tRange[1])
         [obs, p0, p1] = data.getScores(["obs", pvar0, pvar1])
      obs = Metric.within(obs, tRange)
      p = p1 - p0
      if(np.mean(p) == 0):
         return np.nan
      return np.mean(obs) / np.mean(p)

   def label(self, data):
      return "Ratio of marginal probs: Pobs/Pfcst"


class SpreadSkillDiff(Metric):
   _description = "Difference between spread and skill in %"
   _perfectScore = 0
   _orientation = 0

   def computeCore(self, data, tRange):
      import scipy.stats
      [obs, fcst, spread] = data.getScores(["obs", "fcst", "spread"])
      if(len(obs) <= 1):
         return np.nan
      rmse = np.sqrt(np.mean((obs - fcst) ** 2))
      spread = np.mean(spread) / 2.563103
      return 100 * (spread / rmse - 1)

   def name(self):
      return "Spread-skill difference"

   def label(self, data):
      return "Spread-skill difference (%)"


class Within(Metric):
   _min = 0
   _max = 100
   _description = "The percentage of forecasts within some"\
         " error bound (use -r)"
   _defaultBinType = "below"
   _reqThreshold = True
   _supThreshold = True
   _perfectScore = 100
   _orientation = -1

   def computeCore(self, data, tRange):
      [obs, fcst] = data.getScores(["obs", "fcst"])
      diff = abs(obs - fcst)
      return np.mean(self.within(diff, tRange)) * 100

   def name(self):
      return "Within"

   def label(self, data):
      return "% of forecasts"


# Mean y conditioned on x
# For a given range of x-values, what is the average y-value?
class Conditional(Metric):
   _orientation = 0
   _reqThreshold = True
   _supThreshold = True

   def __init__(self, x="obs", y="fcst", func=np.mean):
      self._x = x
      self._y = y
      self._func = func

   def computeCore(self, data, tRange):
      [obs, fcst] = data.getScores([self._x, self._y])
      I = np.where(self.within(obs, tRange))[0]
      if(len(I) == 0):
         return np.nan
      return self._func(fcst[I])


# Mean x when conditioned on x. Average x-value that is within a given range.
# The reason the y-variable is added is to ensure that the same data is used
# for this metric as for the Conditional metric.
class XConditional(Metric):
   _orientation = 0
   _reqThreshold = True
   _supThreshold = True

   def __init__(self, x="obs", y="fcst"):
      self._x = x
      self._y = y

   def computeCore(self, data, tRange):
      [obs, fcst] = data.getScores([self._x, self._y])
      I = np.where(self.within(obs, tRange))[0]
      if(len(I) == 0):
         return np.nan
      return np.median(obs[I])


# Counts how many values of a specific variable is within the threshold range
# Not a real metric.
class Count(Metric):
   _orientation = 0
   _reqThreshold = True
   _supThreshold = True

   def __init__(self, x):
      self._x = x

   def computeCore(self, data, tRange):
      values = data.getScores(self._x)
      I = np.where(self.within(values, tRange))[0]
      if(len(I) == 0):
         return np.nan
      return len(I)


class Bs(Metric):
   _min = 0
   _max = 1
   _description = "Brier score"
   _reqThreshold = True
   _supThreshold = True
   _perfectScore = 0
   _orientation = 1
   _reference = "Glenn W. Brier, 1950: Verification of forecasts expressed in terms of probability. Mon. Wea. Rev., 78, 1-3."

   def __init__(self, numBins=10):
      self._edges = np.linspace(0, 1.0001, numBins)

   def computeCore(self, data, tRange):
      # Compute probabilities based on thresholds
      p0 = 0
      p1 = 1
      if(tRange[0] != -np.inf and tRange[1] != np.inf):
         var0 = data.getPvar(tRange[0])
         var1 = data.getPvar(tRange[1])
         [obs, p0, p1] = data.getScores(["obs", var0, var1])
      elif(tRange[0] != -np.inf):
         var0 = data.getPvar(tRange[0])
         [obs, p0] = data.getScores(["obs", var0])
      elif(tRange[1] != np.inf):
         var1 = data.getPvar(tRange[1])
         [obs, p1] = data.getScores(["obs", var1])
      obsP = self.within(obs, tRange)
      p = p1 - p0  # Prob of obs within range
      bs = np.nan * np.zeros(len(p), 'float')

      # Split into bins and compute Brier score on each bin
      for i in range(0, len(self._edges) - 1):
         I = np.where((p >= self._edges[i]) & (p < self._edges[i + 1]))[0]
         if(len(I) > 0):
            bs[I] = (np.mean(p[I]) - obsP[I]) ** 2
      return Util.nanmean(bs)

   @staticmethod
   def getP(data, tRange):
      p0 = 0
      p1 = 1
      if(tRange[0] != -np.inf and tRange[1] != np.inf):
         var0 = data.getPvar(tRange[0])
         var1 = data.getPvar(tRange[1])
         [obs, p0, p1] = data.getScores(["obs", var0, var1])
      elif(tRange[0] != -np.inf):
         var0 = data.getPvar(tRange[0])
         [obs, p0] = data.getScores(["obs", var0])
      elif(tRange[1] != np.inf):
         var1 = data.getPvar(tRange[1])
         [obs, p1] = data.getScores(["obs", var1])

      obsP = Metric.within(obs, tRange)
      p = p1 - p0  # Prob of obs within range
      return [obsP, p]

   @staticmethod
   def getQ(data, tRange):
      p0 = 0
      p1 = 1
      var = data.getQvar(tRange[0])
      [obs, q] = data.getScores(["obs", var])

      return [obs, q]

   def label(self, data):
      return "Brier score"


class Bss(Metric):
   _min = 0
   _max = 1
   _description = "Brier skill score"
   _reqThreshold = True
   _supThreshold = True
   _perfectScore = 1
   _orientation = 1

   def __init__(self, numBins=10):
      self._edges = np.linspace(0, 1.0001, numBins)

   def computeCore(self, data, tRange):
      [obsP, p] = Bs.getP(data, tRange)
      bs = np.nan * np.zeros(len(p), 'float')
      for i in range(0, len(self._edges) - 1):
         I = np.where((p >= self._edges[i]) & (p < self._edges[i + 1]))[0]
         if(len(I) > 0):
            bs[I] = (np.mean(p[I]) - obsP[I]) ** 2
      bs = Util.nanmean(bs)
      bsunc = np.mean(obsP) * (1 - np.mean(obsP))
      if(bsunc == 0):
         bss = np.nan
      else:
         bss = (bsunc - bs) / bsunc
      return bss

   def label(self, data):
      return "Brier skill score"


class BsRel(Metric):
   _min = 0
   _max = 1
   _description = "Brier score, reliability term"
   _reqThreshold = True
   _supThreshold = True
   _perfectScore = 0
   _orientation = 1

   def __init__(self, numBins=11):
      self._edges = np.linspace(0, 1.0001, numBins)

   def computeCore(self, data, tRange):
      [obsP, p] = Bs.getP(data, tRange)

      # Break p into bins, and comute reliability
      bs = np.nan * np.zeros(len(p), 'float')
      for i in range(0, len(self._edges) - 1):
         I = np.where((p >= self._edges[i]) & (p < self._edges[i + 1]))[0]
         if(len(I) > 0):
            meanObsI = np.mean(obsP[I])
            bs[I] = (np.mean(p[I]) - meanObsI) ** 2
      return Util.nanmean(bs)

   def label(self, data):
      return "Brier score, reliability term"


class BsUnc(Metric):
   _min = 0
   _max = 1
   _description = "Brier score, uncertainty term"
   _reqThreshold = True
   _supThreshold = True
   _perfectScore = None
   _orientation = 1

   def computeCore(self, data, tRange):
      [obsP, p] = Bs.getP(data, tRange)
      meanObs = np.mean(obsP)
      bs = meanObs * (1 - meanObs)
      return bs

   def label(self, data):
      return "Brier score, uncertainty term"


class BsRes(Metric):
   _min = 0
   _max = 1
   _description = "Brier score, resolution term"
   _reqThreshold = True
   _supThreshold = True
   _perfectScore = 1
   _orientation = 1

   def __init__(self, numBins=10):
      self._edges = np.linspace(0, 1.0001, numBins)

   def computeCore(self, data, tRange):
      [obsP, p] = Bs.getP(data, tRange)
      bs = np.nan * np.zeros(len(p), 'float')
      meanObs = np.mean(obsP)
      for i in range(0, len(self._edges) - 1):
         I = np.where((p >= self._edges[i]) & (p < self._edges[i + 1]))[0]
         if(len(I) > 0):
            meanObsI = np.mean(obsP[I])
            bs[I] = (meanObsI - meanObs) ** 2
      return Util.nanmean(bs)

   def label(self, data):
      return "Brier score, resolution term"


class QuantileScore(Metric):
   _min = 0
   _description = "Quantile score. Requires quantiles to be stored"\
                  "(e.g q10, q90...).  Use -x to set which quantiles to use."
   _reqThreshold = True
   _supThreshold = True
   _perfectScore = 0
   _orientation = -1

   def computeCore(self, data, tRange):
      [obs, q] = Bs.getQ(data, tRange)
      qs = np.nan * np.zeros(len(q), 'float')
      v = q - obs
      qs = v * (tRange[0] - (v < 0))
      return np.mean(qs)


class Ign0(Metric):
   _description = "Ignorance of the binary probability based on threshold"
   _reqThreshold = True
   _supThreshold = True
   _orientation = -1

   def computeCore(self, data, tRange):
      [obsP, p] = Bs.getP(data, tRange)

      I0 = np.where(obsP == 0)[0]
      I1 = np.where(obsP == 1)[0]
      ign = -np.log2(p)
      ign[I0] = -np.log2(1 - p[I0])
      return np.mean(ign)

   def label(self, data):
      return "Binary Ignorance"


class Spherical(Metric):
   _description = "Spherical probabilistic scoring rule for binary events"
   _reqThreshold = True
   _supThreshold = True
   _max = 1
   _min = 0
   _orientation = -1

   def computeCore(self, data, tRange):
      [obsP, p] = Bs.getP(data, tRange)

      I0 = np.where(obsP == 0)[0]
      I1 = np.where(obsP == 1)[0]
      sp = p / np.sqrt(p ** 2 + (1 - p) ** 2)
      sp[I0] = (1 - p[I0]) / np.sqrt((p[I0]) ** 2 + (1 - p[I0]) ** 2)

      return np.mean(sp)

   def label(self, data):
      return "Spherical score"


# Metrics based on 2x2 contingency table for a given threshold
class Contingency(Metric):
   _min = 0
   _max = 1
   _defaultAxis = "threshold"
   _reqThreshold = True
   _supThreshold = True
   _usingQuantiles = False

   @staticmethod
   def getAxisFormatter(self, data):
      from matplotlib.ticker import ScalarFormatter
      return ScalarFormatter()

   def label(self, data):
      return self.name()

   def computeCore(self, data, tRange):
      [obs, fcst] = data.getScores(["obs", "fcst"])
      return self.computeObsFcst(obs, fcst, tRange)

   # convert a range of quantiles to thresholds, for example converting
   # [10%, 50%] of some precip values to [5 mm, 25 mm]
   def _quantileToThreshold(self, values, tRange):
      sorted = np.sort(values)
      qRange = [-np.inf, np.inf]
      for i in range(0, 1):
         if(not np.isinf(abs(tRange[i]))):
            qRange[i] = np.percentile(sorted, tRange[i] * 100)
      return qRange

   def computeObsFcst(self, obs, fcst, tRange):
      if(tRange is None):
         Util.error("Metric " + self.getClassName() +
               " requires '-r <threshold>'")
      value = np.nan
      if(len(fcst) > 0):
         # Compute frequencies
         if(self._usingQuantiles):
            fcstSort = np.sort(fcst)
            obsSort = np.sort(obs)
            fRange = self._quantileToThreshold(fcstSort, tRange)
            oRange = self._quantileToThreshold(obsSort, tRange)
            a = np.ma.sum((self.within(fcst, fRange)) &
                  (self.within(obs, oRange)))  # Hit
            b = np.ma.sum((self.within(fcst, fRange)) &
                  (self.within(obs, oRange) == 0))  # FA
            c = np.ma.sum((self.within(fcst, fRange) == 0) &
                  (self.within(obs, oRange)))  # Miss
            d = np.ma.sum((self.within(fcst, fRange) == 0) &
                  (self.within(obs, oRange) == 0))  # CR
         else:
            a = np.ma.sum((self.within(fcst, tRange)) &
                  (self.within(obs, tRange)))  # Hit
            b = np.ma.sum((self.within(fcst, tRange)) &
                  (self.within(obs, tRange) == 0))  # FA
            c = np.ma.sum((self.within(fcst, tRange) == 0) &
                  (self.within(obs, tRange)))  # Miss
            d = np.ma.sum((self.within(fcst, tRange) == 0) &
                  (self.within(obs, tRange) == 0))  # CR
         value = self.calc(a, b, c, d)
         if(np.isinf(value)):
            value = np.nan

      return value

   def name(self):
      return self.description()


class Ets(Contingency):
   _description = "Equitable threat score"
   _perfectScore = 1
   _orientation = 1

   def calc(self, a, b, c, d):
      N = a + b + c + d
      ar = (a + b) / 1.0 / N * (a + c)
      if(a + b + c - ar == 0):
         return np.nan
      return (a - ar) / 1.0 / (a + b + c - ar)

   def name(self):
      return "ETS"


class Dscore(Contingency):
   _description = "Generalized discrimination score"
   _perfectScore = 1
   _orientation = 1
   _reference = "Simon J. Mason and Andreas P. Weigel, 2009: A Generic Forecast Verification Framework for Administrative Purposes. Mon. Wea. Rev., 137, 331-349."
   _max = 1
   _min = 0

   def calc(self, a, b, c, d):
      N = a + b + c + d
      num = a*d + 0.5*(a*b + c*d)
      denom = (a + c) * (b + d)
      if(denom == 0):
         return np.nan
      return num / denom

   def name(self):
      return "Discrimination"


class Threat(Contingency):
   _description = "Threat score"
   _perfectScore = 1
   _orientation = 1

   def calc(self, a, b, c, d):
      if(a + b + c == 0):
         return np.nan
      return a / 1.0 / (a + b + c)


class Pc(Contingency):
   _description = "Proportion correct"
   _perfectScore = 1
   _orientation = 1

   def calc(self, a, b, c, d):
      return (a + d) / 1.0 / (a + b + c + d)


class Diff(Contingency):
   _description = "Difference between false alarms and misses"
   _min = -1
   _max = 1
   _perfectScore = 0
   _orientation = 0

   def calc(self, a, b, c, d):
      return (b - c) / 1.0 / (b + c)


class Edi(Contingency):
   _description = "Extremal dependency index"
   _perfectScore = 1
   _orientation = 1
   _reference = "Christopher A. T. Ferro and David B. Stephenson, 2011: Extremal Dependence Indices: Improved Verification Measures for Deterministic Forecasts of Rare Binary Events. Wea. Forecasting, 26, 699-713."

   def calc(self, a, b, c, d):
      N = a + b + c + d
      F = b / 1.0 / (b + d)
      H = a / 1.0 / (a + c)
      if(H == 0 or F == 0):
         return np.nan
      denom = (np.log(H) + np.log(F))
      if(denom == 0):
         return np.nan
      return (np.log(F) - np.log(H)) / denom

   def name(self):
      return "EDI"


class Sedi(Contingency):
   _description = "Symmetric extremal dependency index"
   _perfectScore = 1
   _orientation = 1
   _reference = Edi.reference()

   def calc(self, a, b, c, d):
      N = a + b + c + d
      F = b / 1.0 / (b + d)
      H = a / 1.0 / (a + c)
      if(F == 0 or F == 1 or H == 0 or H == 1):
         return np.nan
      denom = np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H)
      if(denom == 0):
         return np.nan
      num = np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)
      return num / denom

   def name(self):
      return "SEDI"


class Eds(Contingency):
   _description = "Extreme dependency score"
   _min = None
   _perfectScore = 1
   _orientation = 1
   _reference = "Stephenson, D. B., B. Casati, C. A. T. Ferro, and C. A.  Wilson, 2008: The extreme dependency score: A non-vanishing measure for forecasts of rare events. Meteor. Appl., 15, 41-50."

   def calc(self, a, b, c, d):
      N = a + b + c + d
      H = a / 1.0 / (a + c)
      p = (a + c) / 1.0 / N
      if(H == 0 or p == 0):
         return np.nan
      denom = (np.log(p) + np.log(H))
      if(denom == 0):
         return np.nan
      return (np.log(p) - np.log(H)) / denom

   def name(self):
      return "EDS"


class Seds(Contingency):
   _description = "Symmetric extreme dependency score"
   _min = None
   _perfectScore = 1
   _orientation = 1

   def calc(self, a, b, c, d):
      N = a + b + c + d
      H = a / 1.0 / (a + c)
      p = (a + c) / 1.0 / N
      q = (a + b) / 1.0 / N
      if(q == 0 or H == 0):
         return np.nan
      denom = np.log(p) + np.log(H)
      if(denom == 0):
         return np.nan
      return (np.log(q) - np.log(H)) / (np.log(p) + np.log(H))

   def name(self):
      return "SEDS"


class BiasFreq(Contingency):
   _max = None
   _description = "Bias frequency (number of fcsts / number of obs)"
   _perfectScore = 1
   _orientation = 0

   def calc(self, a, b, c, d):
      if(a + c == 0):
         return np.nan
      return 1.0 * (a + b) / (a + c)


class Hss(Contingency):
   _max = None
   _description = "Heidke skill score"
   _perfectScore = 1
   _orientation = 1

   def calc(self, a, b, c, d):
      denom = ((a + c) * (c + d) + (a + b) * (b + d))
      if(denom == 0):
         return np.nan
      return 2.0 * (a * d - b * c) / denom


class BaseRate(Contingency):
   _description = "Base rate"
   _perfectScore = None
   _orientation = 0

   def calc(self, a, b, c, d):
      if(a + b + c + d == 0):
         return np.nan
      return (a + c) / 1.0 / (a + b + c + d)


class Or(Contingency):
   _description = "Odds ratio"
   _max = None
   _perfectScore = None  # Should be infinity
   _orientation = 1

   def calc(self, a, b, c, d):
      if(b * c == 0):
         return np.nan
      return (a * d) / 1.0 / (b * c)


class Lor(Contingency):
   _description = "Log odds ratio"
   _max = None
   _perfectScore = None  # Should be infinity
   _orientation = 1

   def calc(self, a, b, c, d):
      if(a * d == 0 or b * c == 0):
         return np.nan
      return np.log((a * d) / 1.0 / (b * c))


class YulesQ(Contingency):
   _description = "Yule's Q (Odds ratio skill score)"
   _perfectScore = 1
   _orientation = 1

   def calc(self, a, b, c, d):
      if(a * d + b * c == 0):
         return np.nan
      return (a * d - b * c) / 1.0 / (a * d + b * c)


class Kss(Contingency):
   _description = "Hanssen-Kuiper skill score"
   _perfectScore = 1
   _orientation = 1
   _reference = "Hanssen , A., W. Kuipers, 1965: On the relationship between the frequency of rain and various meteorological parameters. - Meded. Verh. 81, 2-15."

   def calc(self, a, b, c, d):
      if((a + c) * (b + d) == 0):
         return np.nan
      return (a * d - b * c) * 1.0 / ((a + c) * (b + d))


class Hit(Contingency):
   _description = "Hit rate (a.k.a. probability of detection)"
   _perfectScore = 1
   _orientation = 1

   def calc(self, a, b, c, d):
      if(a + c == 0):
         return np.nan
      return a / 1.0 / (a + c)


class Miss(Contingency):
   _description = "Miss rate"
   _perfectScore = 0
   _orientation = -1

   def calc(self, a, b, c, d):
      if(a + c == 0):
         return np.nan
      return c / 1.0 / (a + c)


# Fraction of non-events that are forecasted as events
class Fa(Contingency):
   _description = "False alarm rate"
   _perfectScore = 0
   _orientation = -1

   def calc(self, a, b, c, d):
      if(b + d == 0):
         return np.nan
      return b / 1.0 / (b + d)


# Fraction of forecasted events that are false alarms
class Far(Contingency):
   _description = "False alarm ratio"
   _perfectScore = 0
   _orientation = -1

   def calc(self, a, b, c, d):
      if(a + b == 0):
         return np.nan
      return b / 1.0 / (a + b)
