import inspect
import matplotlib.dates
import matplotlib.ticker
import sys
import datetime
import numpy as np
import calendar

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


def get_time_axes():
    return [axis[1]() for axis in get_all() if hasattr(axis[1], "compute_from_times")]


def get_leadtime_axes():
    return [axis[1]() for axis in get_all() if hasattr(axis[1], "compute_from_leadtimes")]
    return [verif.axis.Leadtime(), verif.axis.Leadtimeday()]


class Axis(object):
    """ Represents axes that data can be arranged by

    Attributes:
       is_continuous (bool): Does this axis represent a notion that is
          continuous (does it make sense to connect dots with a line on a
          graph)? For example, Location is not a continuous axis.
       is_location_like (bool): Does this axis related to the notion of location?
       is_time_like (bool): Does this axis have anything to do with the notion of time?
       formatter (matplotlib.ticker.Formatter): What format should this axis
          have? Returns an mpl Formatter
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
    fmt = "%Y-%m-%d %H:%M:%S"

    def formatter(self, variable):
        return matplotlib.dates.DateFormatter('\n%Y-%m-%d')

    def label(self, variable):
        return "Date"


class Leadtime(Axis):
    """ Forecast lead-time """

    def label(self, variable):
        return "Lead time (h)"

    def compute_from_leadtimes(self, leadtimes):
        return leadtimes


class Leadtimeday(Axis):
    """ Forecast lead-time aggregated over a day"""

    def label(self, variable):
        return "Lead time (day)"

    def compute_from_leadtimes(self, leadtimes):
        return np.array([int(d/24) for d in leadtimes])


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

    def formatter(self, variable):
        return matplotlib.dates.DateFormatter('\n%Y')

    def label(self, variable):
        return "Year"

    def compute_from_times(self, times):
        dts = [datetime.datetime.utcfromtimestamp(i) for i in times]
        dts = [d.replace(month=1, day=1, hour=0, minute=0, second=0) for d in dts]
        return np.array([calendar.timegm(dt.timetuple()) for dt in dts])


class Month(Axis):
    is_time_like = True
    fmt = "%Y/%m"

    def formatter(self, variable):
        return matplotlib.dates.DateFormatter('%b\n%Y')

    def label(self, variable):
        return "Month"

    def compute_from_times(self, times):
        dts = [datetime.datetime.utcfromtimestamp(i) for i in times]
        dts = [d.replace(day=1, hour=0, minute=0, second=0) for d in dts]
        return np.array([calendar.timegm(dt.timetuple()) for dt in dts])


class Week(Axis):
    is_time_like = True
    fmt = "%Y/%U"

    def formatter(self, variable):
        return matplotlib.dates.DateFormatter('%Y-%U')

    def label(self, variable):
        return "Week"

    def compute_from_times(self, times):
        dts = [datetime.datetime.utcfromtimestamp(i) for i in times]
        dts = [d.replace(hour=0, minute=0, second=0) for d in dts]
        # Reset datetime such that it is for the first day of the week
        # That is subtract the day of the week from the date
        dts = [d - datetime.timedelta(days=d.weekday()) for d in dts]
        return np.array([calendar.timegm(dt.timetuple()) for dt in dts])


class Timeofday(Axis):
    def label(self, variable):
        return "Time of day"

    def compute_from_times(self, times):
        return (times % 86400) / 3600


class Dayofyear(Axis):
    def label(self, variable):
        return "Day of year"

    def compute_from_times(self, times):
        dts = [datetime.datetime.utcfromtimestamp(i) for i in times]
        dts = [d.replace(year=2000) for d in dts]
        return np.array([(x - datetime.datetime(year=2000, month=1, day=1)).days + 1 for x in dts])


class Day(Axis):
    is_time_like = True
    fmt = "%Y/%m/%d"

    def formatter(self, variable):
        return matplotlib.dates.DateFormatter('\n%Y-%m-%d')

    def label(self, variable):
        return "Day"

    def compute_from_times(self, times):
        dts = [datetime.datetime.utcfromtimestamp(i) for i in times]
        dts = [d.replace(hour=0, minute=0, second=0) for d in dts]
        return np.array([calendar.timegm(dt.timetuple()) for dt in dts])


class Dayofmonth(Axis):
    def label(self, variable):
        return "Day of month"

    def compute_from_times(self, times):
        return np.array([datetime.datetime.utcfromtimestamp(i).day for i in times])


class Monthofyear(Axis):
    def label(self, variable):
        return "Month of year"

    def compute_from_times(self, times):
        return np.array([datetime.datetime.utcfromtimestamp(i).month for i in times])


class Obs(Axis):
    def formatter(self, variable):
        return variable.formatter

    def label(self, variable):
        return "Observed " + variable.name + " (" + variable.units + ")"


class Fcst(Axis):
    def formatter(self, variable):
        return variable.formatter

    def label(self, variable):
        return "Forecasted " + variable.name + " (" + variable.units + ")"


class Threshold(Axis):
    def formatter(self, variable):
        return variable.formatter

    def label(self, variable):
        return variable.name + " (" + variable.units + ")"
