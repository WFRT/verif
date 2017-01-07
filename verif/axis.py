import inspect
import matplotlib.dates
import matplotlib.ticker
import sys

import verif.util


def get_all():
   """
   Returns a dictionary of all axis classes where the key is the class
   name (string) and the value is the class object
   """
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return temp


def get(name):
   """ Returns an instance of an object with the given class name """
   axes = get_all()
   a = None
   for axis in axes:
      if name == axis[0].lower():
         a = axis[1]()
   if a is None:
      verif.util.error("No axis by name '%s'" % name)
   return a


class Axis(object):
   """ Represents axes that data can be arranged by

   Attributes:
      is_continuous (bool): Does this axis represent a notion that is
         continuous (does it make sense to connect dots with a line on a
         graph)? For example, Location is not a continuous axis.
      is_location_like (bool): Does this axis related to the notion of location?
      is_time_like (bool): Does this axis have anything to do with the notion of time?
      formatter (matplotlib.ticker.Formatter): What format should this axis
         have? Returns an mpl Formatter Note the date formatters are never
         retrieved from here, since mpl.gca().xaxis_date() is used instead
   """
   is_continuous = True
   is_location_like = False
   is_time_like = False
   fmt = "%f"

   @classmethod
   def name(cls):
      name = cls.__name__
      return name

   def label(self, variable):
      """ Returns an appropriate string for labeling an axis on a plot """
      raise NotImplementedError()

   def formatter(self, variable):
      """ How should ticks be generated for this axis? """
      return matplotlib.ticker.ScalarFormatter()

   def __eq__(self, other):
      return self.__class__ == other.__class__

   def __ne__(self, other):
      return not self.__eq__(other)

   def __hash__(self):
      # TODO
      return 1


class Time(Axis):
   """ Forecast initialization time """
   is_time_like = True
   fmt = "%Y-%m-%d"

   def formatter(self, variable):
      return matplotlib.dates.DateFormatter('\n%Y-%m-%d')

   def label(self, variable):
      return "Date"


class Leadtime(Axis):
   """ Forecast lead-time """

   def label(self, variable):
      return "Lead time (h)"


class Location(Axis):
   is_continuous = False
   is_location_like = True

   def label(self, variable):
      return "Location"


class Lat(Axis):
   is_continuous = False
   is_location_like = True

   def label(self, variable):
      return "Latitude ($^o$)"


class Lon(Axis):
   is_continuous = False
   is_location_like = True

   def label(self, variable):
      return "Longitude ($^o$)"


class Elev(Axis):
   is_continuous = False
   is_location_like = True

   def label(self, variable):
      return "Elevation (m)"


class All(Axis):
   """ No aggregation done """
   is_continuous = False


class No(Axis):
   """ Entire 3D array is serialized """
   is_continuous = False


class Year(Axis):
   is_time_like = True
   fmt = "%Y"
   formatter = matplotlib.dates.DateFormatter('\n%Y')

   def label(self, variable):
      return "Year"


class Month(Axis):
   is_time_like = True
   fmt = "%Y/%m"
   formatter = matplotlib.dates.DateFormatter('\n%Y-%m')

   def label(self, variable):
      return "Month"


class Week(Axis):
   is_time_like = True
   fmt = "%Y/%U"
   formatter = matplotlib.dates.DateFormatter('\n%Y-%U')

   def label(self, variable):
      return "Week"


class DayOfMonth(Axis):
   is_time_like = True

   def label(self, variable):
      return "Day of month"


class DayOfYear(Axis):
   is_time_like = True

   def label(self, variable):
      return "Day of year"


class MonthOfYear(Axis):
   is_time_like = True

   def label(self, variable):
      return "Month of year"


class Threshold(Axis):
   def formatter(self, variable):
      return variable.formatter

   def label(self, variable):
      return variable.name + " (" + variable.units + ")"
