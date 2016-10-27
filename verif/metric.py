import numpy as np
import verif.util
import sys
import inspect
import verif.axis
import verif.aggregator


def get_all():
   """
   Returns a dictionary of all metric classes where the key is the class
   name (string) and the value is the class object
   """
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return temp


def get_all_deterministic():
   """ Like get_all, except only return deterministic metric classes """
   metrics = [metric for metric in get_all() if issubclass(metric[1], verif.metric.Deterministic)]
   return metrics


def get(name):
   """ Returns an instance of an object with the given class name """
   metrics = get_all()
   m = None
   for metric in metrics:
      if(name == metric[0].lower() and metric[1].is_valid()):
         m = metric[1]()
   return m


class Metric(object):
   """
   Class to compute a score

   Class attributes:
   requires_threshold:  Does this metric require thresholds in order to be computable?
   supports_threshold:  Does it make sense to use '-x threshold' with this metric?
   orientation:         1 for a positively oriented score (higher values are better),
                        -1 for negative, and 0 for all others
   """
   # Overload these variables
   min = None  # Minimum value this metric can produce
   max = None  # Maximum value this mertic can produce
   default_axis = verif.axis.Offset  # If no axis is specified, use this axis as default
   default_bin_type = None
   requires_threshold = False  # Does this metric require thresholds?
   supports_threshold = False  # Does this metric support thresholds?
   experimental = False  # Is this metric not fully tested yet?
   perfect_score = None
   aggregator = verif.aggregator.Mean
   supports_aggregator = False  # Does this metric use self.aggregator?
   orientation = 0
   # Information about metric. The y-axis label is controlled by self.label()
   # Also, self.name() is the name of the metric

   # A short one-liner describing the metric. This will show up in the
   # main verif documentation.
   description = ""
   # A longer description. This will show up in the documentation when a
   # specific metric is chosen.
   long = None
   reference = None  # A string with an academic reference

   def compute(self, data, input_index, axis, threshold_range):
      """
      Compute the score

      Arguments:
      data              use get_scores([metric1, metric2...]) to get data data has already been
                        configured to only retrieve data along a certain dimension
      threshold_range   [lowerThreshold, upperThreshold]

      Returns:
      scores            A numpy array of one score for each slice along axis
      """
      size = data.get_axis_size(axis)
      scores = np.zeros(size, 'float')
      # Loop through axis indices
      for axis_index in range(0, size):
         x = self.compute_core(data, input_index, axis, axis_index, threshold_range)
         scores[axis_index] = x
      return scores

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      """ Computes the score for a given slice

      Arguments:
      input_index       Which input index to compute the result for
      axis              Along which axis to compute for
      axis_index        What slice along the axis
      threshold_range         
      
      Returns a scalar value representing the score for the slice
      """
      raise NotImplementedError()

   @classmethod
   def is_valid(cls):
      """ Is this a valid metric that can be initialized? """
      return cls.summary() is not ""

   @classmethod
   def help(cls):
      s = cls.description
      if(cls.orientation is not 0):
         s = s + "\n" + verif.util.green("Orientation: ")
         if(cls.orientation == 1):
            s = s + "Positive"
         elif(cls.orientation == -1):
            s = s + "Negative"
         else:
            s = s + "None"
      if(cls.perfect_score is not None):
         s = s + "\n" + verif.util.green("Perfect score: ") + str(cls.perfect_score)
      if(cls.min is not None):
         s = s + "\n" + verif.util.green("Minimum value: ") + str(cls.min)
      if(cls.max is not None):
         s = s + "\n" + verif.util.green("Maximum value: ") + str(cls.max)
      if(cls.long is not None):
         s = s + "\n" + verif.util.green("Description: ") + cls.long
      if(cls.reference is not None):
         s = s + "\n" + verif.util.green("Reference: ") + cls.reference
      return s

   @classmethod
   def summary(cls):
      desc = cls.description
      if(desc == ""):
         return ""
      extra = ""
      # if(cls.experimental):
      #    extra = " " + verif.util.experimental() + "."
      if(cls.supports_aggregator):
         extra = " Supports -ct."
      if(cls.perfect_score is not None):
         extra = extra + " " + "Perfect score " + str(cls.perfect_score) + "."
      return desc + "." + extra

   @classmethod
   def get_class_name(cls):
      name = cls.__name__
      return name

   def label(self, data):
      """ What is an appropriate y-axis label for this metric? Override this if the metric does not
      have the same units as the forecast variable """
      return self.name() + " (" + data.variable.units + ")"

   def name(self):
      """ Cannot be a classmethod, since it might use self.aggregator """
      return self.get_class_name()


class Deterministic(Metric):
   """ Class for scores that are based on observations and deterministic forecasts only """

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obs, fcst] = data.get_scores([verif.field.Obs, verif.field.Deterministic], input_index, axis, axis_index)
      assert(obs.shape[0] == fcst.shape[0])
      return self.compute_from_obs_fcst(obs, fcst)

   def compute_from_obs_fcst(self, obs, fcst):
      """
      Compute the score using only the observations and forecasts

      obs      1D numpy array of observations
      fcst     1D numpy array of forecasts

      obs and fcst must have the same length, but may contain nan values
      """

      # Remove missing values
      I = np.where((np.isnan(obs) | np.isnan(fcst)) == 0)[0]
      obs = obs[I]
      fcst = fcst[I]
      if(obs.shape[0] > 0):
         return self._compute_from_obs_fcst(obs, fcst)
      else:
         return np.nan

   def _compute_from_obs_fcst(self, obs, fcst):
      """
      Subclass must implement this function. Preconditions for obs and fcst:
          - obs and fcst are the same length
          - length >= 1
          - no missing values
       """
      raise NotImplementedError()

class Standard(Metric):
   # aux: When reading the score, also pull values for 'aux' to ensure
   # only common data points are returned
   def __init__(self, name, aux=None):
      self._name = name
      self._aux = aux

   def compute_core(self, data, input_index, axis, axis_index, tRange):
      if(self._aux is not None):
         [values, aux] = data.get_scores([self._name, self._aux], input_index,
               axis, axis_index)
      else:
         values = data.get_scores(self._name, input_index, axis, axis_index)
      return self.aggregator(values)

   def name(self):
      return self.aggregator.name.title() + " of " + self._name


# Note: This cannot be a subclass of Deterministic, since we don't want
# to remove obs for which the forecasts are missing. Same for Fcst.
class Obs(Metric):
   description = "Observed value"
   supports_aggregator = True
   orientation = 0

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      obs = data.get_scores(verif.field.Obs, input_index, axis, axis_index)[0]
      return self.aggregator(obs)

   def name(self):
      return self.aggregator.name.title() + " of observation"


class Fcst(Metric):
   description = "Forecasted value"
   supports_aggregator = True
   orientation = 0

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      fcst = data.get_scores(verif.field.Deterministic, input_index, axis, axis_index)[0]
      return self.aggregator(fcst)

   def name(self):
      return self.aggregator.name.title() + " of forecast"


class Mae(Deterministic):
   description = "Mean absolute error"
   min = 0
   perfect_score = 0
   supports_aggregator = True

   def _compute_from_obs_fcst(self, obs, fcst):
      return self.aggregator(abs(obs - fcst))

   def name(self):
      return "MAE"


class Bias(Deterministic):
   description = "Bias"
   perfect_score = 0
   supports_aggregator = True
   orientation = 0

   def _compute_from_obs_fcst(self, obs, fcst):
      return self.aggregator(obs - fcst)


class Ef(Deterministic):
   description = "Exeedance fraction: percentage of times that forecasts"\
                  " > observations"
   min = 0
   max = 100
   perfect_score = 50
   orientation = 0

   def _compute_from_obs_fcst(self, obs, fcst):
      Nfcst = np.sum(obs < fcst)
      return Nfcst / 1.0 / len(fcst) * 100

   def name(self):
      return "Exceedance fraction"

   def label(self, data):
      return "% times fcst > obs"


class StdError(Deterministic):
   min = 0
   description = "Standard error (i.e. RMSE if forecast had no bias)"
   perfect_score = 0
   orientation = -1

   def _compute_from_obs_fcst(self, obs, fcst):
      bias = np.mean(obs - fcst)
      return np.mean((obs - fcst - bias) ** 2) ** 0.5

   def name(self):
      return "Standard error"


class Rmse(Deterministic):
   min = 0
   description = "Root mean squared error"
   perfect_score = 0
   orientation = -1

   def _compute_from_obs_fcst(self, obs, fcst):
      return np.mean((obs - fcst) ** 2) ** 0.5

   def name(self):
      return "RMSE"


class Rmsf(Deterministic):
   min = 0
   description = "Root mean squared factor"
   perfect_score = 1
   orientation = 0

   def _compute_from_obs_fcst(self, obs, fcst):
      return np.exp(np.mean((np.log(fcst / obs)) ** 2) ** 0.5)

   def name(self):
      return "RMSF"


class Crmse(Deterministic):
   min = 0
   description = "Centered root mean squared error (RMSE without bias)"
   perfect_score = 0
   orientation = -1

   def _compute_from_obs_fcst(self, obs, fcst):
      bias = np.mean(obs) - np.mean(fcst)
      return np.mean((obs - fcst - bias) ** 2) ** 0.5

   def name(self):
      return "CRMSE"


class Cmae(Deterministic):
   min = 0
   description = "Cube-root mean absolute cubic error"
   perfect_score = 0
   orientation = -1

   def _compute_from_obs_fcst(self, obs, fcst):
      return (np.mean(abs(obs ** 3 - fcst ** 3))) ** (1.0 / 3)

   def name(self):
      return "CMAE"


class Nsec(Deterministic):
   min = 0
   description = "Nash-Sutcliffe efficiency coefficient"
   perfect_score = 1
   orientation = 1
   max = 1

   def _compute_from_obs_fcst(self, obs, fcst):
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
   description = "Alpha index"
   perfect_score = 0
   orientation = -1
   max = 2
   min = 0

   def _compute_from_obs_fcst(self, obs, fcst):
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
   min = 0
   description = "Linear error in probability space"
   perfect_score = 0
   orientation = -1

   def _compute_from_obs_fcst(self, obs, fcst):
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
   description = "Degree of mass balance (obs/fcst)"
   perfect_score = 1
   orientation = 0

   def _compute_from_obs_fcst(self, obs, fcst):
      return np.mean(obs) / np.mean(fcst)

   def name(self):
      return "Degree of mass balance (obs/fcst)"


class Mbias(Deterministic):
   description = "Multiplicative bias (obs/fcst)"
   perfect_score = 1
   orientation = 0

   def _compute_from_obs_fcst(self, obs, fcst):
      return (np.mean(obs) / np.mean(fcst))

   def name(self):
      return self.description

   def label(self, data):
      return self.description


class Corr(Deterministic):
   min = 0  # Technically -1, but values below 0 are not as interesting
   max = 1
   description = "Correlation between obesrvations and forecasts"
   perfect_score = 1
   orientation = 1

   def _compute_from_obs_fcst(self, obs, fcst):
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
   min = 0  # Technically -1, but values below 0 are not as interesting
   max = 1
   description = "Rank correlation between obesrvations and forecasts"
   perfect_score = 1
   orientation = 1

   def _compute_from_obs_fcst(self, obs, fcst):
      import scipy.stats
      if(len(obs) <= 1):
         return np.nan
      return scipy.stats.spearmanr(obs, fcst)[0]

   def name(self):
      return "Rank correlation"

   def label(self, data):
      return "Rank correlation"


class KendallCorr(Deterministic):
   min = 0  # Technically -1, but values below 0 are not as interesting
   max = 1
   description = "Kendall correlation between obesrvations and forecasts"
   perfect_score = 1
   orientation = 1

   def _compute_from_obs_fcst(self, obs, fcst):
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


# Returns all PIT values
class Pit(Metric):
   min = 0
   max = 1
   orientation = 0

   def __init__(self, name="pit"):
      self._name = name

   def label(self, data):
      return "PIT"

   def compute(self, data, threshold_range):
      x0 = data.get_variable().get_x0()
      x1 = data.get_variable().get_x1()
      if(x0 is None and x1 is None):
         [pit] = data.get_scores([self._name])
      else:
         [obs, pit] = data.get_scores(["obs", self._name])
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
   min = 0
   # max = 1
   perfect_score = 1
   description = "Deviation of the PIT histogram"
   orientation = -1

   def __init__(self, numBins=11):
      self._metric = Pit()
      self._bins = np.linspace(0, 1, numBins)

   def label(self, data):
      return "PIT histogram deviation"

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      pit = self._metric.compute(data, input_index, axis, axis_index, threshold_range)
      pit = pit[np.isnan(pit) == 0]

      nb = len(self._bins) - 1
      D = self.deviation(pit, nb)
      D0 = self.expected_deviation(pit, nb)
      dev = D / D0
      return dev

   def name(self):
      return "PIT deviation factor"

   @staticmethod
   def expected_deviation(values, numBins):
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
   def deviation_std(values, numBins):
      if(len(values) == 0 or numBins == 0):
         return np.nan
      n = len(values)
      p = 1.0 / numBins
      numPerBinStd = np.sqrt(n * p * (1 - p))
      std = numPerBinStd / n
      return std

   # What reduction in ignorance is possible by calibrating the PIT-histogram?
   @staticmethod
   def ignorance_potential(values, numBins):
      if(len(values) == 0 or numBins == 0):
         return np.nan
      x = np.linspace(0, 1, numBins + 1)
      n = np.histogram(values, x)[0]
      n = n * 1.0 / sum(n)
      expected = 1.0 / numBins
      ign = np.sum(n * np.log2(n / expected)) / sum(n)
      return ign


class MarginalRatio(Metric):
   min = 0
   description = "Ratio of marginal probability of obs to marginal" \
         " probability of fcst. Use -r."
   perfect_score = 1
   requires_threshold = True
   supports_threshold = True
   default_axis = verif.axis.Threshold
   experimental = True
   orientation = 0

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      if(np.isinf(threshold_range[0])):
         pvar = data.get_p_var(threshold_range[1])
         [obs, p1] = data.get_scores([verif.fields.Obs, pvar], input_index, axis, axis_index)
         p0 = 0 * p1
      elif(np.isinf(threshold_range[1])):
         pvar = data.get_p_var(threshold_range[0])
         [obs, p0] = data.get_scores([verif.fields.Obs, pvar], input_index,
               axis, axis_index)
         p1 = 0 * p0 + 1
      else:
         pvar0 = data.get_p_var(threshold_range[0])
         pvar1 = data.get_p_var(threshold_range[1])
         [obs, p0, p1] = data.get_scores([verif.fields.Obs, pvar0, pvar1],
               input_index, axis, axis_index)
      obs = verif.util.within(obs, threshold_range)
      p = p1 - p0
      if(np.mean(p) == 0):
         return np.nan
      return np.mean(obs) / np.mean(p)

   def label(self, data):
      return "Ratio of marginal probs: Pobs/Pfcst"


class SpreadSkillDiff(Metric):
   description = "Difference between spread and skill in %"
   perfect_score = 0
   orientation = 0

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      import scipy.stats
      [obs, fcst, spread] = data.get_scores([verif.fields.Obs,
         verif.fields.Deterministic, verif.fields.Spread], input_index, axis,
         axis_index)
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
   """ Can't be a subclass of Deterministic, because it depends on threshold
   """
   min = 0
   max = 100
   description = "The percentage of forecasts within some"\
         " error bound (use -r)"
   default_bin_type = "below"
   requires_threshold = True
   supports_threshold = True
   perfect_score = 100
   orientation = -1

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obs, fcst] = data.get_scores([verif.field.Obs,
         verif.field.Deterministic], input_index, axis, axis_index)
      diff = abs(obs - fcst)
      return np.mean(verif.util.within(diff, threshold_range)) * 100

   def name(self):
      return "Within"

   def label(self, data):
      return "% of forecasts"


# Mean y conditioned on x
# For a given range of x-values, what is the average y-value?
class Conditional(Metric):
   orientation = 0
   requires_threshold = True
   supports_threshold = True

   def __init__(self, x=verif.field.Obs, y=verif.field.Deterministic, func=np.mean):
      self._x = x
      self._y = y
      self._func = func

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obs, fcst] = data.get_scores([self._x, self._y], input_index, axis, axis_index)
      I = np.where(verif.util.within(obs, threshold_range))[0]
      if(len(I) == 0):
         return np.nan
      return self._func(fcst[I])


# Mean x when conditioned on x. Average x-value that is within a given range.
# The reason the y-variable is added is to ensure that the same data is used
# for this metric as for the Conditional metric.
class XConditional(Metric):
   orientation = 0
   requires_threshold = True
   supports_threshold = True

   def __init__(self, x=verif.field.Obs, y=verif.field.Deterministic):
      self._x = x
      self._y = y

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obs, fcst] = data.get_scores([self._x, self._y], input_index, axis,
            axis_index)
      I = np.where(verif.util.within(obs, threshold_range))[0]
      if(len(I) == 0):
         return np.nan
      return np.median(obs[I])


# Counts how many values of a specific variable is within the threshold range
# Not a real metric.
class Count(Metric):
   orientation = 0
   requires_threshold = True
   supports_threshold = True

   def __init__(self, x):
      self._x = x

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      values = data.get_scores(self._x, input_index, axis, axis_index)
      I = np.where(verif.util.within(values, threshold_range))[0]
      if(len(I) == 0):
         return np.nan
      return len(I)


class Quantile(Metric):
   min = 0
   max = 1

   def __init__(self, quantile):
      self._quantile = quantile

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      var = data.get_q_var(self._quantile)
      scores = data.get_scores(var, input_index, axis, axis_index)
      return verif.util.nanmean(scores)


class Bs(Metric):
   min = 0
   max = 1
   description = "Brier score"
   requires_threshold = True
   supports_threshold = True
   perfect_score = 0
   orientation = 1
   reference = "Glenn W. Brier, 1950: Verification of forecasts expressed in terms of probability. Mon. Wea. Rev., 78, 1-3."

   def __init__(self, numBins=10):
      self._edges = np.linspace(0, 1.0001, numBins)

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      # Compute probabilities based on thresholds
      p0 = 0
      p1 = 1
      if(threshold_range[0] != -np.inf and threshold_range[1] != np.inf):
         var0 = data.get_p_var(threshold_range[0])
         var1 = data.get_p_var(threshold_range[1])
         [obs, p0, p1] = data.get_scores([verif.field.Obs, var0, var1], input_index,
               axis, axis_index)
      elif(threshold_range[0] != -np.inf):
         var0 = data.get_p_var(threshold_range[0])
         [obs, p0] = data.get_scores([verif.field.Obs, var0], input_index, axis,
               axis_index)
      elif(threshold_range[1] != np.inf):
         var1 = data.get_p_var(threshold_range[1])
         [obs, p1] = data.get_scores([verif.field.Obs, var1], input_index, axis,
               axis_index)
      obsP = verif.util.within(obs, threshold_range)
      p = p1 - p0  # Prob of obs within range
      bs = np.nan * np.zeros(len(p), 'float')

      # Split into bins and compute Brier score on each bin
      for i in range(0, len(self._edges) - 1):
         I = np.where((p >= self._edges[i]) & (p < self._edges[i + 1]))[0]
         if(len(I) > 0):
            bs[I] = (np.mean(p[I]) - obsP[I]) ** 2
      return verif.util.nanmean(bs)

   @staticmethod
   def get_p(data, input_index, axis, axis_index, threshold_range):
      p0 = 0
      p1 = 1
      if(threshold_range[0] != -np.inf and threshold_range[1] != np.inf):
         var0 = data.get_p_var(threshold_range[0])
         var1 = data.get_p_var(threshold_range[1])
         [obs, p0, p1] = data.get_scores([verif.field.Obs, var0, var1],
               input_index, axis, axis_index)
      elif(threshold_range[0] != -np.inf):
         var0 = data.get_p_var(threshold_range[0])
         [obs, p0] = data.get_scores([verif.field.Obs, var0], input_index,
               axis, axis_index)
      elif(threshold_range[1] != np.inf):
         var1 = data.get_p_var(threshold_range[1])
         [obs, p1] = data.get_scores([verif.field.Obs, var1], input_index,
               axis, axis_index)

      obsP = verif.util.within(obs, threshold_range)
      p = p1 - p0  # Prob of obs within range
      return [obsP, p]

   @staticmethod
   def get_q(data, input_index, axis, axis_index, threshold_range):
      p0 = 0
      p1 = 1
      var = data.get_q_var(threshold_range[0])
      [obs, q] = data.get_scores(["obs", var], input_index, axis, axis_index)

      return [obs, q]

   def label(self, data):
      return "Brier score"


class Bss(Metric):
   min = 0
   max = 1
   description = "Brier skill score"
   requires_threshold = True
   supports_threshold = True
   perfect_score = 1
   orientation = 1

   def __init__(self, numBins=10):
      self._edges = np.linspace(0, 1.0001, numBins)

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obsP, p] = Bs.get_p(data, input_index, axis, axis_index, threshold_range)
      bs = np.nan * np.zeros(len(p), 'float')
      for i in range(0, len(self._edges) - 1):
         I = np.where((p >= self._edges[i]) & (p < self._edges[i + 1]))[0]
         if(len(I) > 0):
            bs[I] = (np.mean(p[I]) - obsP[I]) ** 2
      bs = verif.util.nanmean(bs)
      bsunc = np.mean(obsP) * (1 - np.mean(obsP))
      if(bsunc == 0):
         bss = np.nan
      else:
         bss = (bsunc - bs) / bsunc
      return bss

   def label(self, data):
      return "Brier skill score"


class BsRel(Metric):
   min = 0
   max = 1
   description = "Brier score, reliability term"
   requires_threshold = True
   supports_threshold = True
   perfect_score = 0
   orientation = 1

   def __init__(self, numBins=11):
      self._edges = np.linspace(0, 1.0001, numBins)

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obsP, p] = Bs.get_p(data, input_index, axis, axis_index, threshold_range)

      # Break p into bins, and comute reliability
      bs = np.nan * np.zeros(len(p), 'float')
      for i in range(0, len(self._edges) - 1):
         I = np.where((p >= self._edges[i]) & (p < self._edges[i + 1]))[0]
         if(len(I) > 0):
            meanObsI = np.mean(obsP[I])
            bs[I] = (np.mean(p[I]) - meanObsI) ** 2
      return verif.util.nanmean(bs)

   def label(self, data):
      return "Brier score, reliability term"


class BsUnc(Metric):
   min = 0
   max = 1
   description = "Brier score, uncertainty term"
   requires_threshold = True
   supports_threshold = True
   perfect_score = None
   orientation = 1

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obsP, p] = Bs.get_p(data, input_index, axis, axis_index, threshold_range)
      meanObs = np.mean(obsP)
      bs = meanObs * (1 - meanObs)
      return bs

   def label(self, data):
      return "Brier score, uncertainty term"


class BsRes(Metric):
   min = 0
   max = 1
   description = "Brier score, resolution term"
   requires_threshold = True
   supports_threshold = True
   perfect_score = 1
   orientation = 1

   def __init__(self, numBins=10):
      self._edges = np.linspace(0, 1.0001, numBins)

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obsP, p] = Bs.get_p(data, input_index, axis, axis_index, threshold_range)
      bs = np.nan * np.zeros(len(p), 'float')
      meanObs = np.mean(obsP)
      for i in range(0, len(self._edges) - 1):
         I = np.where((p >= self._edges[i]) & (p < self._edges[i + 1]))[0]
         if(len(I) > 0):
            meanObsI = np.mean(obsP[I])
            bs[I] = (meanObsI - meanObs) ** 2
      return verif.util.nanmean(bs)

   def label(self, data):
      return "Brier score, resolution term"


class QuantileScore(Metric):
   min = 0
   description = "Quantile score. Requires quantiles to be stored"\
                  "(e.g q10, q90...).  Use -x to set which quantiles to use."
   requires_threshold = True
   supports_threshold = True
   perfect_score = 0
   orientation = -1

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obs, q] = Bs.get_q(data, input_index, axis, axis_index, threshold_range)
      qs = np.nan * np.zeros(len(q), 'float')
      v = q - obs
      qs = v * (threshold_range[0] - (v < 0))
      return np.mean(qs)


class Ign0(Metric):
   description = "Ignorance of the binary probability based on threshold"
   requires_threshold = True
   supports_threshold = True
   orientation = -1

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obsP, p] = Bs.get_p(data, input_index, axis, axis_index, threshold_range)

      I0 = np.where(obsP == 0)[0]
      I1 = np.where(obsP == 1)[0]
      ign = -np.log2(p)
      ign[I0] = -np.log2(1 - p[I0])
      return np.mean(ign)

   def label(self, data):
      return "Binary Ignorance"


class Spherical(Metric):
   description = "Spherical probabilistic scoring rule for binary events"
   requires_threshold = True
   supports_threshold = True
   max = 1
   min = 0
   orientation = -1

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obsP, p] = Bs.get_p(data, input_index, axis, axis_index, threshold_range)

      I0 = np.where(obsP == 0)[0]
      I1 = np.where(obsP == 1)[0]
      sp = p / np.sqrt(p ** 2 + (1 - p) ** 2)
      sp[I0] = (1 - p[I0]) / np.sqrt((p[I0]) ** 2 + (1 - p[I0]) ** 2)

      return np.mean(sp)

   def label(self, data):
      return "Spherical score"


class Contingency(Metric):
   """ Metrics based on 2x2 contingency table for a given threshold """
   min = 0
   max = 1
   default_axis = verif.axis.Threshold
   requires_threshold = True
   supports_threshold = True
   _usingQuantiles = False

   @staticmethod
   def get_axis_formatter(self, data):
      from matplotlib.ticker import ScalarFormatter
      return ScalarFormatter()

   def label(self, data):
      return self.name()

   def compute_core(self, data, input_index, axis, axis_index, threshold_range):
      [obs, fcst] = data.get_scores([verif.field.Obs, verif.field.Deterministic], input_index, axis, axis_index)
      return self.compute_from_obs_fcst(obs, fcst, threshold_range)

   def _quantile_to_threshold(self, values, threshold_range):
      """
      convert a range of quantiles to thresholds, for example converting
      [10%, 50%] of some precip values to [5 mm, 25 mm]
      """
      sorted = np.sort(values)
      qRange = [-np.inf, np.inf]
      for i in range(0, 1):
         if(not np.isinf(abs(threshold_range[i]))):
            qRange[i] = np.percentile(sorted, threshold_range[i] * 100)
      return qRange

   def compute_from_obs_fcst(self, obs, fcst, threshold_range):
      """
      Computes the score.

      obs      numpy array of observations
      fcst     numpy array of forecasts
      threshold_range   2-valued list of thresholds (lower, upper)
      """
      if(threshold_range is None):
         verif.util.error("Metric " + self.get_class_name() +
               " requires '-r <threshold>'")
      value = np.nan
      if(len(fcst) > 0):
         # Compute frequencies
         if(self._usingQuantiles):
            fcstSort = np.sort(fcst)
            obsSort = np.sort(obs)
            fRange = self._quantile_to_threshold(fcstSort, threshold_range)
            oRange = self._quantile_to_threshold(obsSort, threshold_range)
            a = np.ma.sum((verif.util.within(fcst, fRange)) &
                  (verif.util.within(obs, oRange)))  # Hit
            b = np.ma.sum((verif.util.within(fcst, fRange)) &
                  (verif.util.within(obs, oRange) == 0))  # FA
            c = np.ma.sum((verif.util.within(fcst, fRange) == 0) &
                  (verif.util.within(obs, oRange)))  # Miss
            d = np.ma.sum((verif.util.within(fcst, fRange) == 0) &
                  (verif.util.within(obs, oRange) == 0))  # CR
         else:
            a = np.ma.sum((verif.util.within(fcst, threshold_range)) &
                  (verif.util.within(obs, threshold_range)))  # Hit
            b = np.ma.sum((verif.util.within(fcst, threshold_range)) &
                  (verif.util.within(obs, threshold_range) == 0))  # FA
            c = np.ma.sum((verif.util.within(fcst, threshold_range) == 0) &
                  (verif.util.within(obs, threshold_range)))  # Miss
            d = np.ma.sum((verif.util.within(fcst, threshold_range) == 0) &
                  (verif.util.within(obs, threshold_range) == 0))  # CR
         value = self.compute_from_abcd(a, b, c, d)
         if(np.isinf(value)):
            value = np.nan

      return value

   def compute_from_obs_fcst_fast(self, obs, fcst, othreshold, fthresholds):
      """
      A quicker way to compute when scores are needed for multiple forecast
      thresholds.

      obs         numpy array of observations
      fcst        numpy array of forecasts
      othreshold  observation threshold
      fthresholds numpy array of forecasts thresholds
      """
      values = np.nan * np.zeros(len(fthresholds), 'float')
      if(len(fcst) > 0):
         for t in range(0, len(fthresholds)):
            fthreshold = fthresholds[t]
            a = np.ma.sum((fcst > fthreshold) & (obs > othreshold))
            b = np.ma.sum((fcst > fthreshold) & (obs <= othreshold))
            c = np.ma.sum((fcst <= fthreshold) & (obs > othreshold))
            d = np.ma.sum((fcst <= fthreshold) & (obs <= othreshold))
            value = self.compute_from_abcd(a, b, c, d)
            if(np.isinf(value)):
               value = np.nan
            values[t] = value
      return values

   def compute_from_obs_fcst_resample(self, obs, fcst, othreshold, fthresholds, N):
      """
      Same as compute_from_obs_fcst_fast, except compute more robust scores by
      resampling (with replacement) using the computed values of a, b, c, d.
      Resample N times (resamples only if N > 1).
      """
      values = np.nan * np.zeros(len(fthresholds), 'float')
      if(len(fcst) > 0):
         for t in range(0, len(fthresholds)):
            fthreshold = fthresholds[t]
            a = np.ma.sum((fcst > fthreshold) & (obs > othreshold))
            b = np.ma.sum((fcst > fthreshold) & (obs <= othreshold))
            c = np.ma.sum((fcst <= fthreshold) & (obs > othreshold))
            d = np.ma.sum((fcst <= fthreshold) & (obs <= othreshold))
            n = a + b + c + d
            np.random.seed(1)
            currValues = np.nan*np.zeros(N, 'float')
            value = 0
            if N > 1:
               for i in range(0, N):
                  aa = np.random.binomial(n, 1.0*a/n)
                  bb = np.random.binomial(n, 1.0*b/n)
                  cc = np.random.binomial(n, 1.0*c/n)
                  dd = np.random.binomial(n, 1.0*d/n)
                  value = value + self.compute_from_abcd(aa, bb, cc, dd)
               value = value / N
            else:
               value = self.compute_from_abcd(a, b, c, d)
            values[t] = value
      return values

   def name(self):
      return self.description()


class A(Contingency):
   description = "Hit"

   def compute_from_abcd(self, a, b, c, d):
      return 1.0 * a / (a + b + c + d)


class B(Contingency):
   description = "False alarm"

   def compute_from_abcd(self, a, b, c, d):
      return 1.0 * b / (a + b + c + d)


class C(Contingency):
   description = "Miss"

   def compute_from_abcd(self, a, b, c, d):
      return 1.0 * c / (a + b + c + d)


class D(Contingency):
   description = "Correct rejection"

   def compute_from_abcd(self, a, b, c, d):
      return 1.0 * d / (a + b + c + d)


class N(Contingency):
   description = "Total cases"
   max = None

   def compute_from_abcd(self, a, b, c, d):
      return a + b + c + d


class Ets(Contingency):
   description = "Equitable threat score"
   perfect_score = 1
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      N = a + b + c + d
      ar = (a + b) / 1.0 / N * (a + c)
      if(a + b + c - ar == 0):
         return np.nan
      return (a - ar) / 1.0 / (a + b + c - ar)

   def name(self):
      return "ETS"


class Dscore(Contingency):
   description = "Generalized discrimination score"
   perfect_score = 1
   orientation = 1
   reference = "Simon J. Mason and Andreas P. Weigel, 2009: A Generic Forecast Verification Framework for Administrative Purposes. Mon. Wea. Rev., 137, 331-349."
   max = 1
   min = 0

   def compute_from_abcd(self, a, b, c, d):
      N = a + b + c + d
      num = a*d + 0.5*(a*b + c*d)
      denom = (a + c) * (b + d)
      if(denom == 0):
         return np.nan
      return num / denom

   def name(self):
      return "Discrimination"


class Threat(Contingency):
   description = "Threat score"
   perfect_score = 1
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      if(a + b + c == 0):
         return np.nan
      return a / 1.0 / (a + b + c)


class Pc(Contingency):
   description = "Proportion correct"
   perfect_score = 1
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      return (a + d) / 1.0 / (a + b + c + d)


class Diff(Contingency):
   description = "Difference between false alarms and misses"
   min = -1
   max = 1
   perfect_score = 0
   orientation = 0

   def compute_from_abcd(self, a, b, c, d):
      return (b - c) / 1.0 / (b + c)


class Edi(Contingency):
   description = "Extremal dependency index"
   perfect_score = 1
   orientation = 1
   reference = "Christopher A. T. Ferro and David B. Stephenson, 2011: Extremal Dependence Indices: Improved Verification Measures for Deterministic Forecasts of Rare Binary Events. Wea. Forecasting, 26, 699-713."

   def compute_from_abcd(self, a, b, c, d):
      N = a + b + c + d
      if b + d == 0 or a + c == 0:
         return np.nan
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
   description = "Symmetric extremal dependency index"
   perfect_score = 1
   orientation = 1
   reference = Edi.reference

   def compute_from_abcd(self, a, b, c, d):
      N = a + b + c + d
      if b + d == 0 or a + c == 0:
         return np.nan
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
   description = "Extreme dependency score"
   min = None
   perfect_score = 1
   orientation = 1
   reference = "Stephenson, D. B., B. Casati, C. A. T. Ferro, and C. A.  Wilson, 2008: The extreme dependency score: A non-vanishing measure for forecasts of rare events. Meteor. Appl., 15, 41-50."

   def compute_from_abcd(self, a, b, c, d):
      N = a + b + c + d
      if a + c == 0:
         return np.nan
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
   description = "Symmetric extreme dependency score"
   min = None
   perfect_score = 1
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      N = a + b + c + d
      if a + c == 0:
         return np.nan
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
   max = None
   description = "Bias frequency (number of fcsts / number of obs)"
   perfect_score = 1
   orientation = 0

   def compute_from_abcd(self, a, b, c, d):
      if(a + c == 0):
         return np.nan
      return 1.0 * (a + b) / (a + c)


class Hss(Contingency):
   max = None
   description = "Heidke skill score"
   perfect_score = 1
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      denom = ((a + c) * (c + d) + (a + b) * (b + d))
      if(denom == 0):
         return np.nan
      return 2.0 * (a * d - b * c) / denom


class BaseRate(Contingency):
   description = "Base rate"
   perfect_score = None
   orientation = 0

   def compute_from_abcd(self, a, b, c, d):
      if(a + b + c + d == 0):
         return np.nan
      return (a + c) / 1.0 / (a + b + c + d)


class Or(Contingency):
   description = "Odds ratio"
   max = None
   perfect_score = None  # Should be infinity
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      if(b * c == 0):
         return np.nan
      return (a * d) / 1.0 / (b * c)


class Lor(Contingency):
   description = "Log odds ratio"
   max = None
   perfect_score = None  # Should be infinity
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      if(a * d == 0 or b * c == 0):
         return np.nan
      return np.log((a * d) / 1.0 / (b * c))


class YulesQ(Contingency):
   description = "Yule's Q (Odds ratio skill score)"
   perfect_score = 1
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      if(a * d + b * c == 0):
         return np.nan
      return (a * d - b * c) / 1.0 / (a * d + b * c)


class Kss(Contingency):
   description = "Hanssen-Kuiper skill score"
   perfect_score = 1
   orientation = 1
   reference = "Hanssen , A., W. Kuipers, 1965: On the relationship between the frequency of rain and various meteorological parameters. - Meded. Verh. 81, 2-15."

   def compute_from_abcd(self, a, b, c, d):
      if((a + c) * (b + d) == 0):
         return np.nan
      return (a * d - b * c) * 1.0 / ((a + c) * (b + d))


class Hit(Contingency):
   description = "Hit rate (a.k.a. probability of detection)"
   perfect_score = 1
   orientation = 1

   def compute_from_abcd(self, a, b, c, d):
      if(a + c == 0):
         return np.nan
      return a / 1.0 / (a + c)


class Miss(Contingency):
   description = "Miss rate"
   perfect_score = 0
   orientation = -1

   def compute_from_abcd(self, a, b, c, d):
      if(a + c == 0):
         return np.nan
      return c / 1.0 / (a + c)


# Fraction of non-events that are forecasted as events
class Fa(Contingency):
   description = "False alarm rate"
   perfect_score = 0
   orientation = -1

   def compute_from_abcd(self, a, b, c, d):
      if(b + d == 0):
         return np.nan
      return b / 1.0 / (b + d)


# Fraction of forecasted events that are false alarms
class Far(Contingency):
   description = "False alarm ratio"
   perfect_score = 0
   orientation = -1

   def compute_from_abcd(self, a, b, c, d):
      if(a + b == 0):
         return np.nan
      return b / 1.0 / (a + b)
