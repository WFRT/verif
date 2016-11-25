import datetime
import re
import calendar
import numpy as np
import sys
import verif.interval
from matplotlib.dates import *
import copy
import matplotlib.pyplot as mpl
import textwrap
import os


def bin(x, y, edges, func=np.mean):
   yy = np.nan*np.zeros(len(edges)-1, 'float')
   xx = (edges[0:-1] + edges[1:]) / 2
   for i in range(0, len(xx)):
      I = np.where((x > edges[i]) & (x <= edges[i + 1]))[0]
      if len(I) > 0:
         yy[i] = func(y[I])
   return xx, yy


def convert_dates(dates):
   numDates = len(dates)
   dates2 = np.zeros([numDates], 'float')
   for i in range(0, numDates):
      year = int(dates[i] / 10000)
      month = int(dates[i] / 100 % 100)
      day = int(dates[i] % 100)
      dates2[i] = date2num(datetime.datetime(year, month, day, 0))
   return dates2


def convert_times(times):
   num_times = len(times)
   times2 = np.zeros([num_times], 'float')
   for i in range(0, num_times):
      dt = datetime.datetime.utcfromtimestamp(times[i])
      times2[i] = date2num(dt)
   return times2


def convert_to_yyyymmdd(dates):
   num_dates = len(dates)
   dates2 = np.zeros([num_dates], 'int')
   for i in range(0, num_dates):
      dates2[i] = int(num2date(dates[i]).strftime("%Y%m%d"))
   return dates2


def date_to_unixtime_slow(date):
   ut = calendar.timegm(datetime.datetime.strptime(str(date), "%Y%m%d").timetuple())
   return ut


def date_to_unixtime(date):
   year = date / 10000
   month = date / 100 % 100
   day = date % 100
   ut = calendar.timegm(datetime.datetime(year, month, day).timetuple())
   return ut

def unixtime_to_date(unixtime):
   dt = datetime.datetime.utcfromtimestamp(int(unixtime))
   date = dt.year * 10000 + dt.month * 100 + dt.day
   return date


def red(text):
   """ Print text in red to the console """
   return "\033[31m" + text + "\033[0m"


def green(text):
   """ Print text in green to the console """
   return "\033[32m" + text + "\033[0m"


def yellow(text):
   """ Print text in yellow to the console """
   return "\033[33m" + text + "\033[0m"


def remove_margin():
   mpl.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)


def experimental():
   return yellow("(experimental)")


def error(message):
   """ Write error message to console and abort """
   print "\033[1;31mError: " + message + "\033[0m"
   sys.exit(1)


def warning(message):
   """ Write a warning message to console """
   print "\033[1;33mWarning: " + message + "\033[0m"


def parse_numbers(numbers, isDate=False):
   """
   Parses numbers from an input string. Recognizes MATLAB syntax, such as:
   3              single numbers
   3,4,5          list of numbers
   3:5            number range
   3:2:12         number range with a step size of 2
   3,4:6,2:5:9,6  combinations

   Aborts if the number cannot be parsed.
   """
   # Check if valid string
   if(any(char not in set('-01234567890.:,') for char in numbers)):
      error("Could not translate '" + numbers + "' into numbers")

   values = list()
   commaLists = numbers.split(',')
   for commaList in commaLists:
      colonList = commaList.split(':')
      if(len(colonList) == 1):
         values.append(float(colonList[0]))
      elif(len(colonList) <= 3):
         start = float(colonList[0])
         step = 1
         if(len(colonList) == 3):
            step = float(colonList[1])
         stepSign = step / abs(step)
         # arange does not include the end point:
         end = float(colonList[-1]) + stepSign * 0.0001
         if(isDate):
            date = min(start, end)
            curr = list()
            while date <= max(start, end):
               curr.append(date)
               date = get_date(date, step)
            values = values + list(curr)
         else:
            values = values + list(np.arange(start, end, step))
      else:
         error("Could not translate '" + numbers + "' into numbers")
      if(isDate):
         for i in range(0, len(values)):
            values[i] = int(values[i])
   return values


def subplot(i, N):
   """ Sets up subplot for index i (starts at 0) out of N """
   [X, Y] = get_subplot_size(N)
   mpl.subplot(Y, X, i + 1)


def get_subplot_size(N):
   Y = 1
   if(N > 4):
      Y = np.ceil(np.sqrt(N) / 1.5)
   X = np.ceil(N / Y)
   return [int(X), int(Y)]


def get_map_resolution(lats, lons):
   dlat = (max(lats) - min(lats))
   dlon = (max(lons) - min(lons))
   scale = max(dlat, dlon)
   if(np.isnan(scale)):
      res = "c"
   elif(scale > 60):
      res = "c"
   elif(scale > 1):
      res = "i"
   elif(scale > 0.001):
      res = "h"
   elif(scale > 0.0001):
      res = "f"
   else:
      res = "c"
   return res


def fill(x, yLower, yUpper, col, alpha=1, zorder=0, hatch=''):
   """
   Fill an area along x, between yLower and yUpper. Both yLower and yUpper most
   correspond to points in x (i.e. be in the same order)
   """
   # Populate a list of non-missing points
   X = list()
   Y = list()
   for i in range(0, len(x)):
      if(not(np.isnan(x[i]) or np.isnan(yLower[i]))):
         X.append(x[i])
         Y.append(yLower[i])
   for i in range(len(x) - 1, -1, -1):
      if(not (np.isnan(x[i]) or np.isnan(yUpper[i]))):
         X.append(x[i])
         Y.append(yUpper[i])
   if(len(X) > 0):
      mpl.fill(X, Y, facecolor=col, alpha=alpha, linewidth=0, zorder=zorder,
            hatch=hatch)


def clean(data):
   data = data[:].astype(float)
   q = copy.deepcopy(data)
   # Remove missing values. Convert to -999 and then back to nan to avoid
   # warning messages when doing <, >, and == comparisons with nan.
   q[np.isnan(q)] = -999
   q[(q == -999) | (q < -1000000) | (q > 1e30)] = np.nan
   return q


def get_date(date, diff):
   """
   Date calculation: Adds 'diff' to 'date'

   date     An integer of the form YYYYMMDD
   diff     Number of days to add to date
   """
   year = int(date / 10000)
   month = int(date / 100 % 100)
   day = int(date % 100)
   date2 = datetime.datetime(year, month, day, 0) + datetime.timedelta(diff)
   return int(date2.strftime('%Y%m%d'))


def nanmean(data, **args):
   return np.ma.filled(np.ma.masked_array(data, np.isnan(data)).mean(**args),
         fill_value=np.nan)


def nanmedian(data, **args):
   I = np.where(np.isnan(data.flatten()) == 0)[0]
   return np.median(data.flatten()[I])


def nanmin(data, **args):
   return np.ma.filled(np.ma.masked_array(data, np.isnan(data)).min(**args),
         fill_value=np.nan)


def nanmax(data, **args):
   return np.ma.filled(np.ma.masked_array(data, np.isnan(data)).max(**args),
         fill_value=np.nan)


def nanstd(data, **args):
   return np.ma.filled(np.ma.masked_array(data, np.isnan(data)).std(**args),
         fill_value=np.nan)


def nanpercentile(data, pers):
   I = np.where(np.isnan(data.flatten()) == 0)[0]
   p = np.nan
   if(len(I) > 0):
      p = np.percentile(data.flatten()[I], pers)
   return p


def numvalid(data, **args):
   I = np.where(np.isnan(data.flatten()) == 0)[0]
   return len(I)


def nprange(data):
   return np.max(data) - np.min(data)


def intersect(list1, list2):
   return list(set(list1) & set(list2))


def get_screen_width():
   """ How many character wide is the console? """
   rows, columns = os.popen('stty size', 'r').read().split()
   columns = int(columns)
   return columns


def get_p_var(threshold):
   return "p%g" % (threshold)


def is_number(s):
   """ Returns true if x is a scalar number """
   try:
      float(s)
      return True
   except ValueError:
      return False


def distance(lat1, lon1, lat2, lon2):
   """
   Computes the great circle distance between two points using the
   haversine formula. Values can be vectors.
   """
   # Convert from degrees to radians
   pi = 3.14159265
   lon1 = lon1 * 2 * pi / 360
   lat1 = lat1 * 2 * pi / 360
   lon2 = lon2 * 2 * pi / 360
   lat2 = lat2 * 2 * pi / 360
   dlon = lon2 - lon1
   dlat = lat2 - lat1
   a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
   c = 2 * np.arcsin(np.sqrt(a))
   distance = 6.367e6 * c
   return distance


def apply_threshold(array, bin_type, threshold, upper_threshold=None):
   """ 
   Use bin_type to turn array into binary values

   array
   bin_type
   threshold
   upper_threshold   Upper threshold (only needed with bin_type = *within*)
   """
   if bin_type == "below":
      array = array < threshold
   elif bin_type == "below=":
      array = array <= threshold
   elif bin_type == "above":
      array = array > threshold
   elif bin_type == "above=":
      array = array >= threshold
   elif upper_threshold is None:
      error("Cannot apply thresholding with bin_type '%s'" % bin_type)
   elif bin_type == "within":
      array = (array > threshold ) & (array < upper_threshold)
   elif bin_type == "within=":
      array = (array > threshold ) & (array <= upper_threshold)
   elif bin_type == "=within":
      array = (array >= threshold ) & (array < upper_threshold)
   elif bin_type == "=within=":
      array = (array >= threshold ) & (array <= upper_threshold)
   return array


def apply_threshold_prob(array, bin_type, upper_array=None):
   """
   Compute the probability of x given bin_type

   array          The CDF at the threshold
   bin_type
   upper_array    The CDF of the upper threshold (only needed with
                  bin_type = *within*)
   """
   if bin_type == "below":
      array = array
   elif bin_type == "below=":
      array = array
   elif bin_type == "above":
      array = 1 - array
   elif bin_type == "above=":
      array = 1 - array
   elif upper_array is None:
      error("Cannoot apply thresholding with bin_type '%s'" % bin_type)
   elif re.compile("within").match(bin_type):
      array = upper_array - array
   return array


def get_threshold_string(bin_type):
   s = ""
   if bin_type == "below":
      s = "<"
   elif bin_type == "below=":
      s = "<="
   elif bin_type == "above":
      s = ">"
   elif bin_type == "above=":
      s = ">="
   else:
      error("Cannot get threshold string for bin_type '%s'" % bin_type)
   return s


def deg2rad(deg):
   """ Convert degrees to radians """
   return deg * np.pi / 180.0


def get_square_axis_limits(xlim, ylim):
   axismin = min(min(ylim), min(xlim))
   axismax = max(max(ylim), max(xlim))
   return [axismin, axismax]


def get_intervals(bin_type, thresholds):
   """ Returns a list of interval objects. If bin_type is *within*, then
   intervals are formed by using each pair of consecutive thresholds. For
   bin_type below* the interval [-np.inf, threshold] is used and for bin_type
   above* the inveral [threshold, np.inf] is used.
   
   Arguments:
   bin_type       one of below, below=, within, =within, within=, =within=,
                  above, above=
   thresholds     numy array of thresholds
   """
   if thresholds is None:
      return [verif.interval.Interval(-np.inf, np.inf, True, True)]

   intervals = list()
   N = len(thresholds)
   if re.compile(".*within.*").match(bin_type):
      N = N - 1
   for i in range(0, N):
      lower_equality = False
      upper_equality = False

      if bin_type in ["below", "below="]:
         lower = -np.inf
         upper = thresholds[i]
      elif bin_type in ["above", "above="]:
         lower = thresholds[i]
         upper = np.inf
      elif bin_type in ["within", "=within", "within=", "=within="]:
         lower = thresholds[i]
         upper = thresholds[i+1]
      else:
         verif.util.error("Unrecognized bintype")
      if bin_type in ["below=", "within=", "=within="]:
         upper_equality = True
      if bin_type in ["above=", "=within", "=within="]:
         lower_equality = True
      intervals.append(verif.interval.Interval(lower, upper, lower_equality, upper_equality))
   return intervals
