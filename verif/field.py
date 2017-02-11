import inspect
import matplotlib.ticker
import numpy as np
import sys
import verif.util


def get_all():
   """
   Returns a dictionary of all field classes where the key is the class
   name (string) and the value is the class object
   """
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return temp


def get(name):
   """ Returns an instance of an object with the given class name """
   fields = get_all()
   f = None
   for field in fields:
      if name == field[0].lower():
         f = field[1]()
   if f is None:
      verif.util.error("No field by name '%s'" % name)
   return f


class Field(object):
   """ Base class representing scalar fields of data that can be retrieved from input files """
   @classmethod
   def name(cls):
      name = cls.__name__
      return name

   def label(self, variable):
      """ Get an appropriate axis label for this field """
      return variable.name

   def units(self, variable):
      """ Get the units of this field """
      return variable.units

   def formatter(self, variable):
      """ What formatter for plotting is appropriate for this field?

      Returns:
         matplotlib.ticker.Formatter: The formatter
      """
      return variable.formatter

   def __eq__(self, other):
      return self.__class__ == other.__class__

   def __ne__(self, other):
      return not self.__eq__(other)

   def __hash__(self):
      # TODO
      return 1


class Obs(Field):
   pass


class Fcst(Field):
   pass


class Ensemble(Field):
   def __init__(self, member):
      self.member = member


class Quantile(Field):
   def __init__(self, quantile):
      self.quantile = quantile

   def name(self):
      return "Quantile(%g)" % self.quantile

   def __eq__(self, other):
      if self.__class__ != other.__class__:
         return False
      return self.quantile == other.quantile


class Threshold(Field):
   def __init__(self, threshold):
      self.threshold = threshold

   def label(self, variable):
      return "Probability"

   def units(self, variable):
      return "%"

   def name(self):
      return "Threshold(%g)" % self.threshold

   def __eq__(self, other):
      if self.__class__ != other.__class__:
         return False
      return self.threshold == other.threshold


class Spread(Field):
   def label(self, variable):
      return "Ensemble spread"


class ObsWindow(Field):
   def label(self, variable):
      return "%s window" % variable.name

   def units(self, variable):
      return "h"


class FcstWindow(Field):
   def label(self, variable):
      return "%s window" % variable.name

   def units(self, variable):
      return "h"


class Other(Field):
   def __init__(self, name):
      self._name = name

   def name(self):
      return "Other(%s)" % self._name

   def units(self, variable):
      return "unknown"

   def label(self, variable):
      return self._name

   def __eq__(self, other):
      if self.__class__ != other.__class__:
         return False
      return self._name == other._name


class Pit(Field):
   def label(self, variable):
      return "Verifying PIT"

   def units(self, variable):
      return "%"

   @staticmethod
   def randomize(obs, pit, x0, x1):
      """
      Randomize PIT values when the observations falls on a discrete mass
      (useful for precipitation). When the obs correspond to a discrete mass,
      the PIT value must be randomized. For example if the obs is 0 mm and the
      CDF at 0 mm is 0.3, then a random number between 0 and 0.3 must be used.

      Arguments:
         obs (np.array): observations
         pit (np.array): pit values
         x0 (float or None): lower threshold (e.g. 0 for precipitation)
         x1 (float or None): upper threshold (e.g. 100 for RH)

      Returns:
         np.array: pit values with randomization
      """
      if(x0 is not None):
         factor = np.random.rand(*obs.shape) * (obs == x0) + (obs != x0)
         pit *= factor
      if(x1 is not None):
         # Same for the upper discrete mass
         factor = np.random.rand(*obs.shape) * (obs == x1) + (obs != x1)
         pit = 1 - pit
         pit *= factor
         pit = 1 - pit
      return pit
