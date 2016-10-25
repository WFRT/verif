import numpy as np
import inspect
import sys
import verif.util


def get_all():
   """
   Returns a dictionary of all aggregator classes where the key is the class
   name (string) and the value is the class object
   """
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return temp


def get(name):
   """ Returns an instance of an object with the given class name """
   aggregators = get_all()
   a = None
   for aggregator in aggregators:
      if name == aggregator[0].lower():
         a = aggregator[1]()
   if a is None and verif.util.is_number(name):
      a = Quantile(float(name))

   return a


class Aggregator(object):
   """
   Base class for aggregating an array

   Usage:
   mean = verif.aggregator.Mean()
   print mean.name
   mean(np.array([1,2,3]))

   name:    A string representing the name of the aggregator
   """
   pass

class Mean(Aggregator):
   name = "mean"
   def __call__(self, array):
      return np.mean(array)


class Median(Aggregator):
   name = "median"
   def __call__(self, array):
      return np.median(array)


class Min(Aggregator):
   name = "min"
   def __call__(self, array):
      return np.min(array)


class Max(Aggregator):
   name = "max"
   def __call__(self, array):
      return np.max(array)


class Std(Aggregator):
   name = "std"
   def __call__(self, array):
      return np.std(array)


class Variance(Aggregator):
   name = "variance"
   def __call__(self, array):
      return np.var(array)


class Iqr(Aggregator):
   name = "iqr"
   def __call__(self, array):
      return np.percentile(array, 75) - np.percentile(array, 25)


class Range(Aggregator):
   name = "range"
   def __call__(self, array):
      return verif.util.nprange(array)


class Count(Aggregator):
   name = "count"
   def __call__(self, array):
      return verif.util.numvalid(array)


class Sum(Aggregator):
   name = "sum"
   def __call__(self, array):
      return np.sum(array)


class Meanabs(Aggregator):
   name = "meanabs"
   def __call__(self, array):
      return verif.util.meanabs(array)


class Absmean(Aggregator):
   name = "absmean"
   def __call__(self, array):
      return np.abs(np.mean(array))


class Quantile(Aggregator):
   name = "quantile"
   def __init__(self, quantile):
      self.quantile = quantile
      if self.quantile < 0 or self.quantile > 1:
         verif.util.error("Quantile must be between 0 and 1")

   def __call__(self, array):
      return np.percentile(array, self.quantile*100)
