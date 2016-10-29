import sys
import inspect
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
   """
   Base class representing scalar fields of data that can be retrieved from
   input files
   """
   @classmethod
   def name(cls):
      name = cls.__name__
      return name

   def __eq__(self, other):
      return self.__class__ == other.__class__

   def units(self, variable):
      return variable.units


class Obs(Field):
   def label(self, variable):
      return variable.name


class Fcst(Field):
   def label(self, variable):
      return variable.name


class Ensemble(Field):
   def __init__(self, member):
      self.member = member

   def label(self, variable):
      return variable.name


class Quantile(Field):
   def __init__(self, quantile):
      self.quantile = quantile

   def label(self, variable):
      return variable.name

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
