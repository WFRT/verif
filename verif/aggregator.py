import numpy as np
import inspect
import sys
import verif.util


def get_all():
   """ Returns a list of all aggregator classes """
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return [i[1] for i in temp if i[0] != "Aggregator"]


def get(name):
   """
   Returns an instance of an object with the given class name
   
   name     The name of the class. Use a number between 0 and 1 to get the
            corresponding quantile aggregator. 
   """
   aggregators = get_all()
   a = None
   for aggregator in aggregators:
      if name == aggregator.name():
         a = aggregator()
   if a is None and verif.util.is_number(name):
      a = Quantile(float(name))

   return a


class Aggregator(object):
   """
   Base class for aggregating an array (i.e. computing a single value from an
   array)

   Usage:
   mean = verif.aggregator.Mean()
   mean(np.array([1,2,3]))

   name:    A string representing the name of the aggregator
   """
   def __hash__(self):
      # TODO
      return 1

   def __eq__(self, other):
      return self.__class__ == other.__class__

   @classmethod
   def name(cls):
      return cls.__name__.lower()

class Mean(Aggregator):
   def __call__(self, array):
      return np.mean(array)


class Median(Aggregator):
   def __call__(self, array):
      return np.median(array)


class Min(Aggregator):
   def __call__(self, array):
      return np.min(array)


class Max(Aggregator):
   def __call__(self, array):
      return np.max(array)


class Std(Aggregator):
   def __call__(self, array):
      return np.std(array)


class Variance(Aggregator):
   def __call__(self, array):
      return np.var(array)


class Iqr(Aggregator):
   def __call__(self, array):
      return np.percentile(array, 75) - np.percentile(array, 25)


class Range(Aggregator):
   def __call__(self, array):
      return verif.util.nprange(array)


class Count(Aggregator):
   def __call__(self, array):
      return verif.util.numvalid(array)


class Sum(Aggregator):
   def __call__(self, array):
      return np.sum(array)


class Meanabs(Aggregator):
   def __call__(self, array):
      return verif.util.meanabs(array)


class Absmean(Aggregator):
   def __call__(self, array):
      return np.abs(np.mean(array))


class Quantile(Aggregator):
   def __init__(self, quantile):
      self.quantile = quantile
      if self.quantile < 0 or self.quantile > 1:
         verif.util.error("Quantile must be between 0 and 1")

   def __call__(self, array):
      return np.percentile(array, self.quantile*100)

   def __eq__(self, other):
      return (self.__class__ == other.__class__) and (self.quantile == other.quantile)
