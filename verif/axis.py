import matplotlib.ticker
import matplotlib.dates
import verif.util


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
   is_continous = False
   is_location_like = True


class Lat(Axis):
   is_continous = False
   is_location_like = True


class Lon(Axis):
   is_continous = False
   is_location_like = True


class Elev(Axis):
   is_continous = False
   is_location_like = True


class LocationId(Axis):
   is_continous = False
   is_location_like = True


class All(Axis):
   is_continous = False


class No(Axis):
   is_continous = False


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
