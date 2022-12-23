from __future__ import absolute_import
import inspect
from . import metric_type
import numpy as np
import sys
import scipy.stats
import verif.aggregator
import verif.axis
import verif.interval
import verif.util


def get_all():
    """
    Returns a dictionary of all metric classes where the key is the class
    name (string) and the value is the class object
    """
    temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    return temp


def get_all_by_type(type):
    """
    Like get_all, except only return metrics that are of a cerrtain
    verif.metric_type
    """
    temp = [m for m in get_all() if m[1].type == type]
    return temp


def get_all_obs_fcst_based():
    """ Like get_all, except only return obs-fcst-based metric classes """
    metrics = [metric for metric in get_all() if issubclass(metric[1], verif.metric.ObsFcstBased)]
    return metrics


def get(name):
    """ Returns an instance of an object with the given class name """
    metrics = get_all()
    m = None
    for metric in metrics:
        if name == metric[0].lower() and metric[1].is_valid():
            m = metric[1]()
    return m


def get_p(data, input_index, axis, axis_index, interval):
    """
    Retrieves and computes forecast probability and verifying observation for
    being inside interval

    Returns:
       obs (np.array): True when observation is inside interval
       p (np.array): True when forecast is inside interval

    """
    p0 = 0
    p1 = 1
    if interval.lower != -np.inf and interval.upper != np.inf:
        var0 = verif.field.Threshold(interval.lower)
        var1 = verif.field.Threshold(interval.upper)
        [obs, p0, p1] = data.get_scores([verif.field.Obs(), var0, var1],
              input_index, axis, axis_index)
    elif interval.lower != -np.inf:
        var0 = verif.field.Threshold(interval.lower)
        [obs, p0] = data.get_scores([verif.field.Obs(), var0], input_index,
              axis, axis_index)
    elif interval.upper != np.inf:
        var1 = verif.field.Threshold(interval.upper)
        [obs, p1] = data.get_scores([verif.field.Obs(), var1], input_index,
              axis, axis_index)

    obsP = interval.within(obs)
    obsP = np.ma.filled(obsP, fill_value=np.nan)
    p = p1 - p0  # Prob of obs within range
    return [obsP, p]


def get_q(data, input_index, axis, axis_index, interval):
    """
    Retrieve forecast quantile and verifying observation

    Returns:
       obs (np.array): True when observation is inside interval
       p (np.array): True when forecast is inside interval

    """
    p0 = 0
    p1 = 1
    var = verif.field.Quantile(interval.lower)
    [obs, q] = data.get_scores([verif.field.Obs(), var], input_index, axis, axis_index)

    return [obs, q]


class Metric(object):
    """ Class to compute a score for a verification metric

    Scores are computed by retrieving information from a verif.data.Data object.
    As data is organized in multiple dimensions, scores are computed for a
    particular verif.axis.Axis. Also data objects have several input files, so
    scores are computed for a particular input.

    The ObsFcstBased class offers a simple way to design a metric that only
    uses observations and forecasts from data.

    Class attributes:
       description (str): A short one-liner describing the metric. This will show
          up in the main verif documentation.
       long (str): A longer description. This will show up in the
          documentation when a specific metric is chosen.
       min (float): Minimum possible value the metric can take on. None if no min.
       max (float): Maximum possible value the metric can take on. None if no max.
       require_threshold_type (str) : What type of thresholds does this metric
          require? One of 'None', 'deterministic', 'threshold', 'quantile'.
       supports_threshold (bool) : Does it make sense to use '-x threshold' with this metric?
       supports_field (bool) : Does it make sense to use '-x obs' or '-x fcst' with this metric?
       orientation (int): 1 for a positively oriented score (higher values are better),
          -1 for negative, and 0 for all others
       reference (str): A string with an academic reference
       supports_aggregator: Does this metric use self.aggregator?
       type (verif.metric_type.MetricType): What type of metric is this?

    To implement a new metric:
       Fill out cls.description and implement compute_core(). The other class
       attributes (see above) are optional.
    """
    # This must be overloaded
    description = None

    # Default values
    long = None
    reference = None
    orientation = 0
    min = None
    max = None
    default_axis = verif.axis.Leadtime()  # If no axis is specified, use this axis as default
    default_bin_type = None
    require_threshold_type = None
    supports_threshold = False
    supports_field = False
    perfect_score = None
    aggregator = verif.aggregator.Mean()
    supports_aggregator = False
    type = verif.metric_type.Deterministic()

    def compute(self, data, input_index, axis, interval):
        """ Compute the score along an axis

        Arguments:
           data (verif.data.Data): data object to get information from
           input_index (int): input index to compute the result for
           axis (verif.axis.Axis): Axis to compute score for for
           interval: Compute score for this interval (only applies to some metrics)

        Returns:
           np.array: A 1D numpy array of one score for each slice along axis
        """
        size = data.get_axis_size(axis)
        scores = np.zeros(size, 'float')

        # Loop through axis indices
        for axis_index in range(0, size):
            x = self.compute_single(data, input_index, axis, axis_index, interval)
            scores[axis_index] = x
        return scores

    def compute_single(self, data, input_index, axis, axis_index, interval):
        """ Computes the score for a given slice

        Arguments:
           data (verif.data.Data): data object to get information from
           input_index (int): input index to compute the result for
           axis (verif.axis.Axis): Axis to compute score for for
           axis_index (int): Slice along the axis
           interval: Compute score for this interval (only applies to some metrics)

        Returns:
           float: Value representing the score for the slice
        """
        raise NotImplementedError()

    def label(self, variable):
        """ What is an appropriate y-axis label for this metric? Override this if
        the metric does not have the same units as the forecast variable """
        return self.name + " (" + variable.units + ")"

    class ClassProperty(property):
        def __get__(self, cls, owner):
            return self.fget.__get__(None, owner)()

    @ClassProperty
    @classmethod
    def name(cls):
        """ Use the class name as default
        """
        return cls.get_class_name()

    @classmethod
    def is_valid(cls):
        """ Is this a valid metric that can be initialized? """
        return cls.description is not None

    @classmethod
    def help(cls):
        s = ""
        if cls.description is not None:
            s = cls.description
        if cls.orientation != 0:
            s = s + "\n" + verif.util.green("Orientation: ")
            if cls.orientation == 1:
                s = s + "Positive (higher values are better)"
            elif cls.orientation == -1:
                s = s + "Negative (lower values are better)"
            else:
                s = s + "None"
        if cls.perfect_score is not None:
            s = s + "\n" + verif.util.green("Perfect score: ") + str(cls.perfect_score)
        if cls.min is not None:
            s = s + "\n" + verif.util.green("Minimum value: ") + str(cls.min)
        if cls.max is not None:
            s = s + "\n" + verif.util.green("Maximum value: ") + str(cls.max)
        if cls.long is not None:
            s = s + "\n" + verif.util.green("Description: ") + cls.long
        if cls.reference is not None:
            s = s + "\n" + verif.util.green("Reference: ") + cls.reference
        return s

    @classmethod
    def get_class_name(cls):
        name = cls.__name__
        return name


class ObsFcstBased(Metric):
    """ Class for scores that are based on observations and deterministic forecasts only """
    type = verif.metric_type.Deterministic()
    supports_field = True

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obs, fcst] = data.get_scores([verif.field.Obs(), verif.field.Fcst()], input_index, axis, axis_index)
        assert(obs.shape[0] == fcst.shape[0])
        if axis == verif.axis.Obs():
            I = np.where(interval.within(obs))
            obs = obs[I]
            fcst = fcst[I]
        elif axis == verif.axis.Fcst():
            I = np.where(interval.within(fcst))
            obs = obs[I]
            fcst = fcst[I]

        return self.compute_from_obs_fcst(obs, fcst, interval)

    def compute_from_obs_fcst(self, obs, fcst, interval=None):
        """ Compute the score using only the observations and forecasts

        obs and fcst must have the same length, but may contain nan values

        Arguments:
           obs (np.array): 1D array of observations
           fcst (np.array): 1D array of forecasts

        Returns:
           float: Value of score
        """

        # Remove missing values
        I = np.where((np.isnan(obs) | np.isnan(fcst)) == 0)[0]
        obs = obs[I]
        fcst = fcst[I]

        if obs.shape[0] > 0:
            return self._compute_from_obs_fcst(obs, fcst)
        else:
            return np.nan

    def _compute_from_obs_fcst(self, obs, fcst):
        """ Compute the score

        Obs and fcst are guaranteed to:
            - have the same length
            - length >= 1
            - no missing values
         """
        raise NotImplementedError()


class FromField(Metric):
    supports_aggregator = True
    supports_field = True

    def __init__(self, field, aux=None):
        """ Compute scores from a field

        Arguments:
           field (verif.field.field): Retrive data from this field
           aux (verif.field.Field): When reading field, also pull values for
              this field to ensure only common data points are returned
        """
        self._field = field
        self._aux = aux

    def compute_single(self, data, input_index, axis, axis_index, interval):
        fields = [self._field]
        axis_pos = None
        if axis == verif.axis.Obs():
            if self._field != verif.field.Obs():
                fields += [verif.field.Obs()]
            axis_pos = len(fields) - 1
        elif axis == verif.axis.Fcst():
            if self._field != verif.field.Fcst():
                fields += [verif.field.Fcst()]
            axis_pos = len(fields) - 1
        if self._aux is not None:
            fields += [self._aux]
        values_array = data.get_scores(fields, input_index, axis, axis_index)
        values = values_array[0]

        # Subset if we have a subsetting axis
        if axis_pos is not None:
            I = np.where(interval.within(values_array[axis_pos]))[0]
            values = values[I]

        return self.aggregator(values)

    def label(self, variable):
        return self.aggregator.name().title() + " of " + self._field.name()


class Obs(FromField):
    """ Retrives the observation

    Note: This cannot be a subclass of ObsFcstBased, since we don't want
    to remove obs for which the forecasts are missing. Same for Fcst.
    """
    type = verif.metric_type.Deterministic()
    name = "Observation"
    description = "Observed value"
    supports_aggregator = True
    orientation = 0

    def __init__(self):
        super(Obs, self).__init__(verif.field.Obs())

    def label(self, variable):
        return self.aggregator.name().title() + " of observation (" + variable.units + ")"


class Fcst(FromField):
    type = verif.metric_type.Deterministic()
    name = "Forecast"
    description = "Forecasted value"
    supports_aggregator = True
    orientation = 0

    def __init__(self):
        super(Fcst, self).__init__(verif.field.Fcst())

    def label(self, variable):
        return self.aggregator.name().title() + " of forecast (" + variable.units + ")"


class Mae(ObsFcstBased):
    description = "Mean absolute error"
    min = 0
    perfect_score = 0
    supports_aggregator = True
    orientation = -1
    name = "Mean absolute error"

    def _compute_from_obs_fcst(self, obs, fcst):
        return self.aggregator(abs(obs - fcst))

    def label(self, variable):
        return "MAE (" + variable.units + ")"


class Bias(ObsFcstBased):
    name = "Bias"
    description = "Bias (forecast - observation)"
    perfect_score = 0
    supports_aggregator = True
    orientation = 0

    def _compute_from_obs_fcst(self, obs, fcst):
        return self.aggregator(fcst - obs)


class Diff(ObsFcstBased):
    name = "Diff"
    description = "Difference in aggregated statistics (agg(forecast) - agg(observation))"
    perfect_score = 0
    supports_aggregator = True
    orientation = 0

    def _compute_from_obs_fcst(self, obs, fcst):
        return self.aggregator(fcst) - self.aggregator(obs)


class Ratio(ObsFcstBased):
    name = "Ratio"
    description = "Ratio of aggregated statistics (agg(forecast) / agg(observation))"
    perfect_score = 1
    supports_aggregator = True
    orientation = 0

    def _compute_from_obs_fcst(self, obs, fcst):
        num = self.aggregator(fcst)
        denum = self.aggregator(obs)

        if denum == 0:
            return np.nan
        return num / denum

    def label(self, variable):
        return "Ratio"


class Ef(ObsFcstBased):
    name = "Exceedance fraction"
    description = "Exeedance fraction: fraction of times that forecasts > observations"
    min = 0
    max = 1
    perfect_score = 0.5
    orientation = 0

    def _compute_from_obs_fcst(self, obs, fcst):
        Nfcst = np.sum(obs < fcst)
        return Nfcst / 1.0 / len(fcst)

    def label(self, variable):
        return "Fraction fcst > obs"


class StdError(ObsFcstBased):
    name = "Standard error"
    description = "Standard error (i.e. RMSE if forecast had no bias)"
    min = 0
    perfect_score = 0
    orientation = -1

    def _compute_from_obs_fcst(self, obs, fcst):
        bias = np.mean(obs - fcst)
        return np.mean((obs - fcst - bias) ** 2) ** 0.5


class Rmse(ObsFcstBased):
    name = "Root mean squared error"
    description = "Root mean squared error"
    min = 0
    perfect_score = 0
    supports_aggregator = True
    orientation = -1

    def _compute_from_obs_fcst(self, obs, fcst):
        return self.aggregator((obs - fcst) ** 2) ** 0.5

    def label(self, variable):
        return "RMSE (" + variable.units + ")"


class Rmsf(ObsFcstBased):
    name = "Root mean squared factor"
    description = "Root mean squared factor"
    min = 0
    perfect_score = 1
    supports_aggregator = True
    orientation = 0

    def _compute_from_obs_fcst(self, obs, fcst):
        return np.exp(self.aggregator((np.log(fcst / obs)) ** 2) ** 0.5)

    def label(self, variable):
        return "RMSF (" + variable.units + ")"


class Cmae(ObsFcstBased):
    name = "Cube-root mean absolute cubic error"
    description = "Cube-root mean absolute cubic error"
    min = 0
    perfect_score = 0
    supports_aggregator = True
    orientation = -1

    def _compute_from_obs_fcst(self, obs, fcst):
        return (self.aggregator(abs(obs ** 3 - fcst ** 3))) ** (1.0 / 3)

    def label(self, variable):
        return "CMAE (" + variable.units + ")"


class Nsec(ObsFcstBased):
    name = "Nash-Sutcliffe efficiency coefficient"
    description = "Nash-Sutcliffe efficiency coefficient"
    min = 0
    max = 1
    perfect_score = 1
    orientation = 1

    def _compute_from_obs_fcst(self, obs, fcst):
        meanobs = np.mean(obs)
        num = np.sum((fcst - obs) ** 2)
        denom = np.sum((obs - meanobs) ** 2)
        if denom == 0:
            return np.nan
        else:
            return 1 - num / denom

    def label(self, variable):
        return "NSEC"


class Alphaindex(ObsFcstBased):
    name = "Alpha index"
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
        if denom == 0:
            return np.nan
        else:
            return 1 - num / denom

    def label(self, variable):
        return self.name


class Leps(ObsFcstBased):
    name = "Linear error in probability space"
    description = "Linear error in probability space"
    min = 0
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
            if len(I > 0):
                qfcst[i] = float(I[0]) / N
            else:
                qfcst[i] = 1
        return np.mean(abs(qfcst - qobs))

    def label(self, variable):
        return "LEPS"


class Dmb(ObsFcstBased):
    name = "Degree of mass balance"
    description = "Degree of mass balance (obs/fcst)"
    perfect_score = 1
    orientation = 0

    def _compute_from_obs_fcst(self, obs, fcst):
        return np.mean(obs) / np.mean(fcst)

    def label(self, variable):
        return self.description


class Mbias(ObsFcstBased):
    name = "Multiplicative bias"
    description = "Multiplicative bias (fcst/obs)"
    perfect_score = 1
    orientation = 0

    def _compute_from_obs_fcst(self, obs, fcst):
        num = np.nanmean(fcst)
        denum = np.nanmean(obs)

        if denum == 0:
            return np.nan
        return num / denum

    def label(self, variable):
        return self.description


class Corr(ObsFcstBased):
    name = "Correlation"
    description = "Correlation between observations and forecasts"
    min = 0  # Technically -1, but values below 0 are not as interesting
    max = 1
    perfect_score = 1
    orientation = 1

    def _compute_from_obs_fcst(self, obs, fcst):
        if len(obs) <= 1:
            return np.nan
        if np.var(fcst) == 0:
            return np.nan
        return np.corrcoef(obs, fcst)[1, 0]

    def label(self, variable):
        return self.name


class RankCorr(ObsFcstBased):
    name = "Rank correlation"
    description = "Rank correlation between observations and forecasts"
    min = 0  # Technically -1, but values below 0 are not as interesting
    max = 1
    perfect_score = 1
    orientation = 1

    def _compute_from_obs_fcst(self, obs, fcst):
        if len(obs) <= 1:
            return np.nan
        return scipy.stats.spearmanr(obs, fcst)[0]

    def label(self, variable):
        return self.name


class KendallCorr(ObsFcstBased):
    name = "Kendall correlation"
    description = "Kendall correlation between observations and forecasts"
    min = 0  # Technically -1, but values below 0 are not as interesting
    max = 1
    perfect_score = 1
    orientation = 1

    def _compute_from_obs_fcst(self, obs, fcst):
        if len(obs) <= 1:
            return np.nan
        if np.var(fcst) == 0:
            return np.nan
        return scipy.stats.kendalltau(obs, fcst)[0]

    def label(self, variable):
        return self.name


class DError(ObsFcstBased):
    name = "Distribution Error"
    description = "Distribution error"
    min = 0
    perfect_score = 0
    supports_aggregator = False
    orientation = -1

    def _compute_from_obs_fcst(self, obs, fcst):
        sortedobs = np.sort(obs)
        sortedfcst = np.sort(fcst)
        return np.mean(np.abs(sortedobs - sortedfcst))


class Pit(Metric):
    """ Retrives the PIT-value corresponding to the observation """
    type = verif.metric_type.Probabilistic()
    name = "Probability integral transform"
    description = "Verifying PIT-value (CDF at observation)"
    supports_aggregator = True
    orientation = 0

    def compute_single(self, data, input_index, axis, axis_index, interval):
        pit = data.get_scores(verif.field.Pit(), input_index, axis, axis_index)
        return self.aggregator(pit)

    def label(self, variable):
        return self.aggregator.name().title() + " of verifying PIT"


class PitHistDev(Metric):
    type = verif.metric_type.Probabilistic()
    name = "PIT histogram deviation factor"
    description = "PIT histogram deviation factor (actual deviation / expected deviation)"
    min = 0
    # max = 1
    perfect_score = 1
    orientation = -1

    def __init__(self, numBins=11, field=verif.field.Pit()):
        self._bins = np.linspace(0, 1, numBins)
        self._field = field

    def compute_single(self, data, input_index, axis, axis_index, interval):
        pit = data.get_scores(self._field, input_index, axis, axis_index)
        nb = len(self._bins) - 1
        D = self.deviation(pit, nb)
        D0 = self.expected_deviation(pit, nb)
        dev = D / D0
        return dev

    def label(self, variable):
        return self.name

    @staticmethod
    def expected_deviation(values, numBins):
        if len(values) == 0 or numBins == 0:
            return np.nan
        return np.sqrt((1.0 - 1.0 / numBins) / (len(values) * numBins))

    @staticmethod
    def deviation(values, numBins):
        if len(values) == 0 or numBins == 0:
            return np.nan
        x = np.linspace(0, 1, numBins + 1)
        n = np.histogram(values, x)[0]
        n = n * 1.0 / sum(n)
        return np.sqrt(1.0 / numBins * np.sum((n - 1.0 / numBins) ** 2))

    @staticmethod
    def deviation_std(values, numBins):
        if len(values) == 0 or numBins == 0:
            return np.nan
        n = len(values)
        p = 1.0 / numBins
        numPerBinStd = np.sqrt(n * p * (1 - p))
        std = numPerBinStd / n
        return std

    # What reduction in ignorance is possible by calibrating the PIT-histogram?
    @staticmethod
    def ignorance_potential(values, numBins):
        if len(values) == 0 or numBins == 0:
            return np.nan
        x = np.linspace(0, 1, numBins + 1)
        n = np.histogram(values, x)[0]
        n = n * 1.0 / sum(n)
        expected = 1.0 / numBins
        ign = np.sum(n * np.log2(n / expected)) / sum(n)
        return ign


class PitHistSlope(Metric):
    type = verif.metric_type.Probabilistic()
    name = "PIT histogram slope"
    description = "Average slope of the PIT histogram. Positive mean too many obs in the higher ranks."
    perfect_score = 0
    orientation = 0

    def __init__(self, numBins=11, field=verif.field.Pit()):
        self._bins = np.linspace(0, 1, numBins)
        self._field = field

    def compute_single(self, data, input_index, axis, axis_index, interval):
        # Create a PIT histogram, then compute the average slope across the bars
        pit = data.get_scores(self._field, input_index, axis, axis_index)
        n = np.histogram(pit, self._bins)[0]
        n = n * 1.0 / sum(n)

        centers = (self._bins[1:] + self._bins[0:-1]) / 2
        dx = np.diff(centers)
        d = np.diff(n) / dx
        return np.mean(d)

    def label(self, variable):
        return self.name


class PitHistShape(Metric):
    type = verif.metric_type.Probabilistic()
    name = "PIT histogram shape"
    description = "Second derivative of the PIT histogram. Negative means U-shaped."
    perfect_score = 0
    orientation = 0

    def __init__(self, numBins=11, field=verif.field.Pit()):
        self._bins = np.linspace(0, 1, numBins)
        self._field = field

    def compute_single(self, data, input_index, axis, axis_index, interval):
        # Create a PIT histogram, then compute the second derivative across the bars
        pit = data.get_scores(self._field, input_index, axis, axis_index)
        n = np.histogram(pit, self._bins)[0]
        n = n * 1.0 / sum(n)

        centers = (self._bins[1:] + self._bins[0:-1]) / 2
        dx = np.diff(centers)
        d = np.diff(n) / dx
        centers2 = (centers[1:] + centers[0:-1]) / 2
        dx2 = np.diff(centers2)
        dd = np.diff(d) / dx2
        return np.mean(dd)

    def label(self, variable):
        return self.name


class MarginalRatio(Metric):
    type = verif.metric_type.Probabilistic()
    name = "Marginal ratio"
    description = "Ratio of marginal probability of obs to marginal" \
          " probability of fcst. Use -r to specify thresholds."
    min = 0
    perfect_score = 1
    require_threshold_type = "threshold"
    supports_threshold = True
    default_axis = verif.axis.Threshold()
    orientation = 0

    def compute_single(self, data, input_index, axis, axis_index, interval):
        if np.isinf(interval.lower):
            pvar = verif.field.Threshold(interval.upper)
            [obs, p1] = data.get_scores([verif.field.Obs(), pvar], input_index, axis, axis_index)
            p0 = 0 * p1
        elif np.isinf(interval.upper):
            pvar = verif.field.Threshold(interval.lower)
            [obs, p0] = data.get_scores([verif.field.Obs(), pvar], input_index,
                  axis, axis_index)
            p1 = 0 * p0 + 1
        else:
            pvar0 = verif.field.Threshold(interval.lower)
            pvar1 = verif.field.Threshold(interval.upper)
            [obs, p0, p1] = data.get_scores([verif.field.Obs(), pvar0, pvar1],
                  input_index, axis, axis_index)
        obs = interval.within(obs)
        p = p1 - p0
        if np.mean(p) == 0:
            return np.nan
        return np.mean(obs) / np.mean(p)

    def label(self, variable):
        return "Ratio of marginal probs: Pobs/Pfcst"


class Within(Metric):
    type = verif.metric_type.Deterministic()
    """ Can't be a subclass of ObsFcstBased, because it depends on threshold
    """
    name = "Within"
    description = "The percentage of forecasts within some error bound. Use -r to specify error bounds"
    min = 0
    max = 100
    default_bin_type = "below"
    require_threshold_type = "threshold"
    supports_threshold = True
    perfect_score = 100
    orientation = 0

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obs, fcst] = data.get_scores([verif.field.Obs(),
           verif.field.Fcst()], input_index, axis, axis_index)

        return self.compute_from_obs_fcst(obs, fcst, interval)

    def compute_from_obs_fcst(self, obs, fcst, interval):
        diff = abs(obs - fcst)
        return np.mean(interval.within(diff)) * 100

    def label(self, variable):
        return "% of forecasts"


class Conditional(Metric):
    """
    Computes the mean y conditioned on x. For a given range of x-values, what is
    the average y-value?
    """
    type = verif.metric_type.Deterministic()
    orientation = 0

    def __init__(self, x=verif.field.Obs(), y=verif.field.Fcst(), func=np.mean):
        self._x = x
        self._y = y
        self._func = func

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obs, fcst] = data.get_scores([self._x, self._y], input_index, axis, axis_index)
        return self.compute_from_obs_fcst(obs, fcst, interval)

    def compute_from_obs_fcst(self, obs, fcst, interval):
        I = np.where(interval.within(obs))[0]
        if len(I) == 0:
            return np.nan
        return self._func(fcst[I])


class XConditional(Metric):
    """
    Mean x when conditioned on x. Average x-value that is within a given range.
    The reason the y-variable is added is to ensure that the same data is used
    for this metric as for the Conditional metric.
    """
    type = verif.metric_type.Deterministic()
    orientation = 0

    def __init__(self, x=verif.field.Obs(), y=verif.field.Fcst(), func=np.median):
        self._x = x
        self._y = y
        self._func = func

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obs, fcst] = data.get_scores([self._x, self._y], input_index, axis, axis_index)
        return self.compute_from_obs_fcst(obs, fcst, interval)

    def compute_from_obs_fcst(self, obs, fcst, interval):
        I = np.where(interval.within(obs))[0]
        if len(I) == 0:
            return np.nan
        return self._func(obs[I])


class Count(Metric):
    """
    Counts how many values of a specific variable is within the threshold range
    Not a real metric.
    """
    type = verif.metric_type.Deterministic()
    orientation = 0

    def __init__(self, x):
        self._x = x

    def compute_single(self, data, input_index, axis, axis_index, interval):
        values = data.get_scores(self._x, input_index, axis, axis_index)
        I = np.where(interval.within(values))[0]
        if len(I) == 0:
            return np.nan
        return len(I)


class Quantile(Metric):
    type = verif.metric_type.Probabilistic()
    min = 0
    max = 1

    def __init__(self, quantile):
        self._quantile = quantile

    def compute_single(self, data, input_index, axis, axis_index, interval):
        var = verif.field.Quantile(self._quantile)
        scores = data.get_scores(var, input_index, axis, axis_index)
        return verif.util.nanmean(scores)


class Bs(Metric):
    type = verif.metric_type.Probabilistic()
    name = "Brier score"
    description = "Brier score"
    min = 0
    max = 1
    default_axis = verif.axis.Threshold()
    require_threshold_type = "threshold"
    supports_threshold = True
    perfect_score = 0
    orientation = -1
    reference = "Glenn W. Brier, 1950: Verification of forecasts expressed in terms of probability. Mon. Wea. Rev., 78, 1-3."

    def compute_single(self, data, input_index, axis, axis_index, interval):
        """ Compute probabilities based on thresholds """
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)
        return self.compute_from_obs_fcst(obsP, p)

    def compute_from_obs_fcst(self, obs, fcst):
        bs = np.nan * np.zeros(len(obs), 'float')
        return np.nanmean((fcst-obs)**2)

    def label(self, variable):
        return self.name


class BsRel(Metric):
    default_axis = verif.axis.Threshold()
    type = verif.metric_type.Probabilistic()
    name = "Brier score, reliability term"
    description = "Brier score, reliability term"
    min = 0
    max = 1
    require_threshold_type = "threshold"
    supports_threshold = True
    perfect_score = 0
    orientation = -1

    def __init__(self, num_edges=11):
        self._edges = np.linspace(0, 1, num_edges)
        self._edges[-1] = 1.001

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)
        return self.compute_from_obs_fcst(obsP, p)

    def compute_from_obs_fcst(self, obs, fcst):
        bs = np.nan * np.zeros(len(fcst), 'float')
        obs_mean = np.mean(obs)
        """
        Break p into bins, and compute reliability. but save each reliability
        value in an array the same size as fcst. In this way we do not need to do
        a weighted average
        """
        for i in range(0, len(self._edges) - 1):
            I = np.where((fcst >= self._edges[i]) & (fcst < self._edges[i + 1]))[0]
            if len(I) > 0:
                obs_mean_I = np.mean(obs[I])
                bs[I] = (fcst[I] - obs_mean_I) ** 2
        return np.nanmean(bs)

    def label(self, variable):
        return self.name


class BsRes(Metric):
    default_axis = verif.axis.Threshold()
    type = verif.metric_type.Probabilistic()
    name = "Brier score, resolution term"
    description = "Brier score, resolution term"
    min = 0
    max = 1
    require_threshold_type = "threshold"
    supports_threshold = True
    perfect_score = 1
    orientation = 1

    def __init__(self, num_edges=11):
        self._edges = np.linspace(0, 1, num_edges)
        self._edges[-1] = 1.001

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)
        return self.compute_from_obs_fcst(obsP, p)

    def compute_from_obs_fcst(self, obs, fcst):
        bs = np.nan * np.zeros(len(fcst), 'float')
        obs_mean = np.mean(obs)
        for i in range(0, len(self._edges) - 1):
            I = np.where((fcst >= self._edges[i]) & (fcst < self._edges[i + 1]))[0]
            if len(I) > 0:
                obs_mean_I = np.mean(obs[I])
                bs[I] = (obs_mean_I - obs_mean) ** 2
        return np.nanmean(bs)

    def label(self, variable):
        return self.name


class BsUnc(Metric):
    default_axis = verif.axis.Threshold()
    type = verif.metric_type.Probabilistic()
    name = "Brier score, uncertainty term"
    description = "Brier score, uncertainty term"
    min = 0
    max = 1
    require_threshold_type = "threshold"
    supports_threshold = True
    perfect_score = None
    orientation = 0

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)
        return self.compute_from_obs_fcst(obsP, p)

    def compute_from_obs_fcst(self, obs, fcst):
        obs_mean = np.mean(obs)
        bsunc = np.nanmean((obs_mean - obs)**2)
        return bsunc

    def label(self, variable):
        return self.name


class Bss(Metric):
    default_axis = verif.axis.Threshold()
    type = verif.metric_type.Probabilistic()
    name = "Brier skill score"
    description = "Brier skill score"
    min = 0
    max = 1
    require_threshold_type = "threshold"
    supports_threshold = True
    perfect_score = 1
    orientation = 1

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)
        return self.compute_from_obs_fcst(obsP, p)

    def compute_from_obs_fcst(self, obs, fcst):
        bs = np.nanmean((fcst - obs)**2)
        obs_mean = np.mean(obs)
        bsunc = np.nanmean((obs_mean - obs)**2)
        if bsunc == 0:
            bss = np.nan
        else:
            bss = (bsunc - bs) / bsunc
        return bss

    def label(self, variable):
        return self.name


class BssRel(Metric):
    default_axis = verif.axis.Threshold()
    type = verif.metric_type.Probabilistic()
    name = "Brier skill score, reliability term"
    description = "Brier skill score, reliability term"
    min = 0
    max = 1
    require_threshold_type = "threshold"
    supports_threshold = True
    perfect_score = 1
    orientation = 1

    def __init__(self, num_edges=11):
        self._edges = np.linspace(0, 1, num_edges)
        self._edges[-1] = 1.001

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)
        return self.compute_from_obs_fcst(obsP, p)

    def compute_from_obs_fcst(self, obs, fcst):
        bsrel = np.nan * np.zeros(len(fcst), 'float')
        obs_mean = np.mean(obs)
        for i in range(0, len(self._edges) - 1):
            I = np.where((fcst >= self._edges[i]) & (fcst < self._edges[i + 1]))[0]
            if len(I) > 0:
                obs_mean_I = np.mean(obs[I])
                bsrel[I] = (fcst[I] - obs_mean_I) ** 2
        bsunc = np.nanmean((obs_mean - obs)**2)
        bsrel = np.nanmean(bsrel)
        if bsunc == 0:
            bss = np.nan
        else:
            bss = bsrel / bsunc
        return bss

    def label(self, variable):
        return self.name


class BssRes(Metric):
    default_axis = verif.axis.Threshold()
    type = verif.metric_type.Probabilistic()
    name = "Brier skill score, resolution term"
    description = "Brier skill score, resolution term"
    min = 0
    max = 1
    require_threshold_type = "threshold"
    supports_threshold = True
    perfect_score = 1
    orientation = 1

    def __init__(self, num_edges=11):
        self._edges = np.linspace(0, 1, num_edges)
        self._edges[-1] = 1.001

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)
        return self.compute_from_obs_fcst(obsP, p)

    def compute_from_obs_fcst(self, obs, fcst):
        bsres = np.nan * np.zeros(len(fcst), 'float')
        obs_mean = np.mean(obs)
        for i in range(0, len(self._edges) - 1):
            I = np.where((fcst >= self._edges[i]) & (fcst < self._edges[i + 1]))[0]
            if len(I) > 0:
                obs_mean_I = np.mean(obs[I])
                bsres[I] = (obs_mean_I - obs_mean) ** 2
        bsres = np.nanmean(bsres)
        bsunc = np.nanmean((obs_mean - obs)**2)
        if bsunc == 0:
            bss = np.nan
        else:
            bss = bsres / bsunc
        return bss

    def label(self, variable):
        return self.name


class QuantileScore(Metric):
    type = verif.metric_type.Probabilistic()
    name = "Quantile score"
    description = "Quantile score. Use -q to set which quantiles to use."
    min = 0
    require_threshold_type = "quantile"
    supports_threshold = True
    perfect_score = 0
    orientation = -1

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obs, q] = get_q(data, input_index, axis, axis_index, interval)
        qs = np.nan * np.zeros(len(q), 'float')
        v = q - obs
        qs = v * (interval.lower - (v < 0))
        return np.mean(qs)

    def label(self, variable):
        return self.name


class Spread(Metric):
    type = verif.metric_type.Probabilistic()
    name = "Spread"
    description = "Spread between two quantiles. Use -q to set which quantiles to use."
    min = 0
    default_bin_type = "within"
    require_threshold_type = "quantile"
    supports_threshold = True
    perfect_score = 0
    orientation = -1

    def compute_single(self, data, input_index, axis, axis_index, interval):
        var0 = verif.field.Quantile(interval.lower)
        var1 = verif.field.Quantile(interval.upper)
        [q0, q1] = data.get_scores([var0, var1], input_index, axis, axis_index)
        return np.mean(q1 - q0)

    def label(self, variable):
        return self.name


class Ign0(Metric):
    type = verif.metric_type.Probabilistic()
    name = "Binary ignorance"
    description = "Ignorance of the binary probability based on threshold"
    require_threshold_type = "threshold"
    supports_threshold = True
    orientation = -1

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)

        I0 = np.where(obsP == 0)[0]
        I1 = np.where(obsP == 1)[0]
        ign = -np.log2(p)
        ign[I0] = -np.log2(1 - p[I0])
        return np.mean(ign)

    def label(self, variable):
        return self.name


class Spherical(Metric):
    type = verif.metric_type.Probabilistic()
    name = "Spherical score"
    description = "Spherical probabilistic scoring rule for binary events"
    require_threshold_type = "threshold"
    supports_threshold = True
    max = 1
    min = 0
    perfect_score = 1
    orientation = 1

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obsP, p] = get_p(data, input_index, axis, axis_index, interval)

        I0 = np.where(obsP == 0)[0]
        I1 = np.where(obsP == 1)[0]
        sp = p / np.sqrt(p ** 2 + (1 - p) ** 2)
        sp[I0] = (1 - p[I0]) / np.sqrt((p[I0]) ** 2 + (1 - p[I0]) ** 2)

        return np.mean(sp)

    def label(self, variable):
        return self.name


class Contingency(Metric):
    """
    Metrics based on 2x2 contingency table for a given interval. Observations
    and forecasts are converted into binary values, that is if they are within
    or not within an interval.
    """
    type = verif.metric_type.Threshold()
    min = 0
    max = 1
    default_axis = verif.axis.Threshold()
    require_threshold_type = "deterministic"
    supports_threshold = True
    _usingQuantiles = False

    def compute_from_abcd(self, a, b, c, d):
        """ Compute the score given the 4 values in the 2x2 contingency table:

        Arguments:
           a (float): Hit
           b (float): False alarm
           c (float): Miss
           d (float): Correct rejection

        Returns:
           float: The score
        """
        raise NotImplementedError()

    def label(self, variable):
        return self.name

    def compute_single(self, data, input_index, axis, axis_index, interval):
        [obs, fcst] = data.get_scores([verif.field.Obs(), verif.field.Fcst()], input_index, axis, axis_index)
        return self.compute_from_obs_fcst(obs, fcst, interval)

    def _quantile_to_threshold(self, values, interval):
        """
        Convert an interval of quantiles to interval thresholds, for example
        converting [10%, 50%] of some precip values to [5 mm, 25 mm]

        Arguments:
           values (np.array): values to compute thresholds for
           interval (verif.interval.Interval): interval of quantiles

        Returns:
           verif.interval.Interval: Interval of thresholds
        """
        sorted = np.sort(values)
        lower = -np.inf
        upper = np.inf
        if not np.isinf(abs(interval.lower)):
            lower = np.percentile(sorted, interval.lower * 100)
        if not np.isinf(abs(interval.lower)):
            upper = np.percentile(sorted, interval.upper * 100)
        return verif.interval.Interval(lower, upper, interval.lower_eq, interval.upper_eq)

    def _compute_abcd(self, obs, fcst, interval, f_interval=None):
        if f_interval is None:
            f_interval = interval
        a = b = c = d = np.nan
        if len(fcst) > 0:
            # Compute frequencies
            if self._usingQuantiles:
                fcstSort = np.sort(fcst)
                obsSort = np.sort(obs)
                f_qinterval = self._quantile_to_threshold(fcstSort, f_interval)
                o_qinterval = self._quantile_to_threshold(obsSort, interval)
                a = np.ma.sum(f_qinterval.within(fcst) & o_qinterval.within(obs))  # Hit
                b = np.ma.sum(f_qinterval.within(fcst) & (o_qinterval.within(obs) == 0))  # FA
                c = np.ma.sum((f_qinterval.within(fcst) == 0) & o_qinterval.within(obs))  # Miss
                d = np.ma.sum((f_qinterval.within(fcst) == 0) & (o_qinterval.within(obs) == 0))  # CR
            else:
                a = np.ma.sum(f_interval.within(fcst) & interval.within(obs))  # Hit
                b = np.ma.sum(f_interval.within(fcst) & (interval.within(obs) == 0))  # FA
                c = np.ma.sum((f_interval.within(fcst) == 0) & interval.within(obs))  # Miss
                d = np.ma.sum((f_interval.within(fcst) == 0) & (interval.within(obs) == 0))  # CR
        return [a, b, c, d]

    def compute_from_obs_fcst(self, obs, fcst, interval, f_interval=None):
        """ Computes the score

        Arguments:
           obs (np.array): array of observations
           fcst (np.array): array of forecasts
           interval (verif.interval.Interval): compute score for this interval
           f_interval (verif.interval.Interval): Use this interval for forecasts.
              If None, then use the same interval for obs and forecasts.

        Returns:
           float: The score
        """
        [a, b, c, d] = self._compute_abcd(obs, fcst, interval, f_interval)

        value = self.compute_from_abcd(a, b, c, d)
        if np.isinf(value):
            value = np.nan

        return value

    def compute_from_obs_fcst_resample(self, obs, fcst, N, interval, f_interval=None):
        """
        Same as compute_from_obs_fcst, except compute more robust scores by
        resampling (with replacement) using the computed values of a, b, c, d.

        Arguments:
           obs (np.array): array of observations
           fcst (np.array): array of forecasts
           N (int): Resample this many times
           interval (verif.interval.Interval): compute score for this interval
           f_interval (verif.interval.Interval): Use this interval for forecasts.
              If None, then use the same interval for obs and forecasts.

        Returns:
           float: The score
        """
        [a, b, c, d] = self._compute_abcd(obs, fcst, interval, f_interval)

        # Resample
        n = a + b + c + d
        np.random.seed(1)
        value = 0
        for i in range(0, N):
            aa = np.random.binomial(n, 1.0*a/n)
            bb = np.random.binomial(n, 1.0*b/n)
            cc = np.random.binomial(n, 1.0*c/n)
            dd = np.random.binomial(n, 1.0*d/n)
            value = value + self.compute_from_abcd(aa, bb, cc, dd)
        value = value / N
        return value

    def label(self, variable):
        return self.name


class A(Contingency):
    name = "Hit"
    description = "Hit"

    def compute_from_abcd(self, a, b, c, d):
        return 1.0 * a / (a + b + c + d)


class B(Contingency):
    name = "False alarm"
    description = "False alarm"

    def compute_from_abcd(self, a, b, c, d):
        return 1.0 * b / (a + b + c + d)


class C(Contingency):
    name = "Miss"
    description = "Miss"

    def compute_from_abcd(self, a, b, c, d):
        return 1.0 * c / (a + b + c + d)


class D(Contingency):
    name = "Correct rejection"
    description = "Correct rejection"

    def compute_from_abcd(self, a, b, c, d):
        return 1.0 * d / (a + b + c + d)


class N(Contingency):
    name = "Total cases"
    description = "Total cases"
    max = None

    def compute_from_abcd(self, a, b, c, d):
        return a + b + c + d


class Ets(Contingency):
    name = "Equitable threat score"
    description = "Equitable threat score"
    perfect_score = 1
    orientation = 1

    def compute_from_abcd(self, a, b, c, d):
        N = a + b + c + d
        ar = (a + b) / 1.0 / N * (a + c)
        if a + b + c - ar == 0:
            return np.nan
        return (a - ar) / 1.0 / (a + b + c - ar)

    def label(self, variable):
        return "ETS"


class FcstRate(Contingency):
    name = "Forecast rate"
    description = "Fractions of forecasts (a + b)"
    perfect_score = None
    orientation = 0

    def compute_from_abcd(self, a, b, c, d):
        return (a + b) / 1.0 / (a + b + c + d)


class Dscore(Contingency):
    name = "Discimination"
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
        if denom == 0:
            return np.nan
        return num / denom


class Threat(Contingency):
    name = "Threat score"
    description = "Threat score"
    perfect_score = 1
    orientation = 1

    def compute_from_abcd(self, a, b, c, d):
        if a + b + c == 0:
            return np.nan
        return a / 1.0 / (a + b + c)


class Pc(Contingency):
    name = "Proportion correct"
    description = "Proportion correct"
    perfect_score = 1
    orientation = 1

    def compute_from_abcd(self, a, b, c, d):
        return (a + d) / 1.0 / (a + b + c + d)


class Edi(Contingency):
    name = "Extremal dependency index"
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
        if H == 0 or F == 0:
            return np.nan
        denom = (np.log(H) + np.log(F))
        if denom == 0:
            return np.nan
        return (np.log(F) - np.log(H)) / denom

    def label(self, variable):
        return "EDI"


class Sedi(Contingency):
    name = "Symmetric extremal dependency index"
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
        if F == 0 or F == 1 or H == 0 or H == 1:
            return np.nan
        denom = np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H)
        if denom == 0:
            return np.nan
        num = np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)
        return num / denom

    def label(self, variable):
        return "SEDI"


class Eds(Contingency):
    name = "Extreme dependency score"
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
        if H == 0 or p == 0:
            return np.nan
        denom = (np.log(p) + np.log(H))
        if denom == 0:
            return np.nan
        return (np.log(p) - np.log(H)) / denom

    def label(self, variable):
        return "EDS"


class Seds(Contingency):
    name = "Symmetric extreme dependency score"
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
        if q == 0 or H == 0:
            return np.nan
        denom = np.log(p) + np.log(H)
        if denom == 0:
            return np.nan
        return (np.log(q) - np.log(H)) / (np.log(p) + np.log(H))

    def label(self, variable):
        return "SEDS"


class BiasFreq(Contingency):
    name = "Bias frequency"
    description = "Bias frequency (number of fcsts / number of obs)"
    max = None
    perfect_score = 1
    orientation = 0

    def compute_from_abcd(self, a, b, c, d):
        if a + c == 0:
            return np.nan
        return 1.0 * (a + b) / (a + c)


class Hss(Contingency):
    name = "Heidke skill score"
    description = "Heidke skill score"
    max = 1
    perfect_score = 1
    orientation = 1

    def compute_from_abcd(self, a, b, c, d):
        denom = ((a + c) * (c + d) + (a + b) * (b + d))
        if denom == 0:
            return np.nan
        return 2.0 * (a * d - b * c) / denom


class BaseRate(Contingency):
    name = "Base rate"
    description = "Base rate: Fraction of observations (a + c)"
    perfect_score = None
    orientation = 0

    def compute_from_abcd(self, a, b, c, d):
        if a + b + c + d == 0:
            return np.nan
        return (a + c) / 1.0 / (a + b + c + d)


class Or(Contingency):
    name = "Odds ratio"
    description = "Odds ratio"
    max = None
    perfect_score = None  # Should be infinity
    orientation = 1

    def compute_from_abcd(self, a, b, c, d):
        if b * c == 0:
            return np.nan
        return (a * d) / 1.0 / (b * c)


class Lor(Contingency):
    name = "Log odds ratio"
    description = "Log odds ratio"
    max = None
    perfect_score = None  # Should be infinity
    orientation = 1

    def compute_from_abcd(self, a, b, c, d):
        if a * d == 0 or b * c == 0:
            return np.nan
        return np.log((a * d) / 1.0 / (b * c))


class YulesQ(Contingency):
    name = "Yule's Q"
    description = "Yule's Q (Odds ratio skill score)"
    perfect_score = 1
    orientation = 1

    def compute_from_abcd(self, a, b, c, d):
        if a * d + b * c == 0:
            return np.nan
        return (a * d - b * c) / 1.0 / (a * d + b * c)


class Kss(Contingency):
    name = "Hanssen-Kuiper skill score"
    description = "Hanssen-Kuiper skill score"
    perfect_score = 1
    orientation = 1
    reference = "Hanssen , A., W. Kuipers, 1965: On the relationship between the frequency of rain and various meteorological parameters. - Meded. Verh. 81, 2-15."

    def compute_from_abcd(self, a, b, c, d):
        if (a + c) * (b + d) == 0:
            return np.nan
        return (a * d - b * c) * 1.0 / ((a + c) * (b + d))


class Hit(Contingency):
    name = "Hit rate"
    description = "Hit rate (a.k.a. probability of detection)"
    perfect_score = 1
    orientation = 1

    def compute_from_abcd(self, a, b, c, d):
        if a + c == 0:
            return np.nan
        return a / 1.0 / (a + c)


class Miss(Contingency):
    name = "Miss rate"
    description = "Miss rate"
    perfect_score = 0
    orientation = -1

    def compute_from_abcd(self, a, b, c, d):
        if a + c == 0:
            return np.nan
        return c / 1.0 / (a + c)


# Fraction of non-events that are forecasted as events
class Fa(Contingency):
    name = "False alarm rate"
    description = "False alarm rate"
    perfect_score = 0
    orientation = -1

    def compute_from_abcd(self, a, b, c, d):
        if b + d == 0:
            return np.nan
        return b / 1.0 / (b + d)


# Fraction of forecasted events that are false alarms
class Far(Contingency):
    name = "False alarm ratio"
    description = "False alarm ratio"
    perfect_score = 0
    orientation = -1

    def compute_from_abcd(self, a, b, c, d):
        if a + b == 0:
            return np.nan
        return b / 1.0 / (a + b)
