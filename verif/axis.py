import sys
import inspect
import matplotlib.ticker
import matplotlib.dates
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
         a = axis[1]
   if a is None:
      verif.util.error("No axis by name '%s'" % name)
   return a


class Axis(object):
   """
   Class attributes:
   is_continuous     Is this axis represent a notion that is continuous? I.e. does it make
                     sense to connect dots with a line on a graph? For example, Location is
                     not a continuous axis.
   is_location_like  Does this axis related to the notion of location?
   is_date_like      Does this axis have anything to do with the notion of date?
   formatter         What format should this axis have? Returns an mpl Formatter
                     Note the date formatters are never retrieved from here, since
                     mpl.gca().xaxis_date() is used instead
   """
   is_continuous = True
   is_location_like = False
   is_date_like = False
   formatter = matplotlib.ticker.ScalarFormatter()

   @classmethod
   def name(cls):
      name = cls.__name__
      return name


class Date(Axis):
   is_date_like = True
   formatter = matplotlib.dates.DateFormatter('\n%Y-%m-%d')


class Offset(Axis):
   pass


class Location(Axis):
   is_continuous = False
   is_location_like = True


class Lat(Axis):
   is_continuous = False
   is_location_like = True


class Lon(Axis):
   is_continuous = False
   is_location_like = True


class Elev(Axis):
   is_continuous = False
   is_location_like = True


class LocationId(Axis):
   is_continuous = False
   is_location_like = True


class All(Axis):
   is_continuous = False


class No(Axis):
   is_continuous = False


class Week(Axis):
   is_date_like = True


class Month(Axis):
   is_date_like = True
   formatter = matplotlib.dates.DateFormatter('\n%Y-%m')


class Year(Axis):
   is_date_like = True
   formatter = matplotlib.dates.DateFormatter('\n%Y')


class DayOfMonth(Axis):
   is_date_like = True


class DayOfYear(Axis):
   is_date_like = True


class MonthOfYear(Axis):
   is_date_like = True


class Threshold(Axis):
   pass
