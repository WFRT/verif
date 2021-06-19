from __future__ import print_function
import matplotlib.dates
import calendar
import copy
import datetime
import matplotlib.pyplot as mpl
import numpy as np
import os
import re
import sys
import textwrap
try:
    from netCDF4 import Dataset as netcdf
except:
    from scipy.io.netcdf import netcdf_file as netcdf

import verif.interval

"""
There are 4 ways to represent time in verif:
   1) unixtime (seconds since 1970). Used internally in verif.data and also
      optionally in input files.
   2) date (YYYYYMMDD). Used on the command-line to specify dates and also
      optionally in input files.
   3) datenum (Some kind of matplotlib way to represent date as integer). Times
      must be in this format to make it easier to put automatic ticks on time axes
   4) datetime (a full date object). Not really used in verif, except when a
      formatted datestring is needed.

Below are functions that convert between the first 3
"""


def date_to_datenum(date):
    """ Converts date in YYYYMMDD format into datenum

    Arguments:
       date (int): date in YYYYMMDD format

    Returns:
       int: datenum value
    """
    year = int(date / 10000)
    month = int(date / 100 % 100)
    day = int(date % 100)
    return matplotlib.dates.date2num(datetime.datetime(year, month, day, 0))


def unixtime_to_datenum(time):
    """ Converts unixtime into datenum

    Arguments:
       time (int): unixtime in seconds since 1970

    Returns:
       int: datenum value
    """
    dt = datetime.datetime.utcfromtimestamp(time)
    return matplotlib.dates.date2num(dt)


def bin(x, y, edges, func=np.nanmean):
    """ Bin the x-values using edges and return the average values

    Arguments:
       x (np.array): x-axis values
       y (np.array): y-axis values
       edges (np.array): bins with these edges
       func: what function to use when binning


    Returns:
       xx (np.array): average values of x in each bin. Length is one less than
          edges
       yy (np.array): average values of y in each bin
    """
    xx = np.nan*np.zeros(len(edges)-1)
    yy = np.nan*np.zeros(len(edges)-1)

    for i in range(len(edges)-1):
        I = np.where((x >= edges[i]) & (x < edges[i+1]))[0]
        if len(I) > 0:
            xx[i] = np.nanmean(x[I])
            yy[i] = func(y[I])

    return xx, yy


def datenum_to_date(datenum):
    """ Converts datenum into YYYYMMDD value

    Arguments:
       date (int): datenum

    Returns:
       int: date in YYYYMMDD
    """
    return int(matplotlib.dates.num2date(datenum).strftime("%Y%m%d"))


def date_to_unixtime(date):
    """ Convert YYYYMMDD to unixtime

    Arguments:
       date (int): YYYYMMDD

    Returns:
       int: unixtime
    """
    year = date // 10000
    month = date // 100 % 100
    day = date % 100
    ut = calendar.timegm(datetime.datetime(year, month, day).timetuple())
    return ut


def date_to_unixtime_slow(date):
    ut = calendar.timegm(datetime.datetime.strptime(str(date), "%Y%m%d").timetuple())
    return ut


def unixtime_to_date(unixtime):
    """ Convert unixtime to YYYYMMDD

    Arguments:
       unixtime (int): unixtime

    Returns:
       int: date in YYYYMMDD
    """
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
    print("\033[1;31mError: " + message + "\033[0m")
    sys.exit(1)


def warning(message):
    """ Write a warning message to console """
    print("\033[1;33mWarning: " + message + "\033[0m")


def parse_numbers(numbers, is_date=False):
    """
    Parses numbers from an input string. Recognizes MATLAB syntax, such as:
    3              single numbers
    3,4,5          list of numbers
    3:5            number range
    3:2:12         number range with a step size of 2
    3,4:6,2:5:9,6  combinations

    Aborts if the number cannot be parsed. Expect round-off errors for values
    below about 1e-4.

    Arguments:
       numbers (str): String of numbers
       is_date (bool): True if values should be interpreted as YYYYMMDD

    Returns:
       list: parsed numbers
    """
    # Check if valid string
    if any(char not in set('-01234567890.:,') for char in numbers):
        error("Could not translate '" + numbers + "' into numbers")

    values = list()
    commaLists = numbers.split(',')
    for commaList in commaLists:
        colonList = commaList.split(':')
        if len(colonList) == 1:
            values.append(float(colonList[0]))
        elif len(colonList) <= 3:
            start = float(colonList[0])
            step = 1
            if len(colonList) == 3:
                step = float(colonList[1])
            if step == 0:
                verif.util.error("Could not parse '%s': Step cannot be 0." % (numbers))
            stepSign = step / abs(step)
            # arange does not include the end point:
            end = float(colonList[-1]) + stepSign * 0.0001
            if is_date:
                date = min(start, end)
                curr = list()
                while date <= max(start, end):
                    curr.append(date)
                    date = get_date(date, step)
                values = values + list(curr)
            else:
                # Note: Values are rounded, to avoid problems with floating point
                # comparison for strings like 0.1:0.1:0.9
                values = values + list(np.round(np.arange(start, end, step), 7))
        else:
            error("Could not translate '" + numbers + "' into numbers")
        if is_date:
            for i in range(0, len(values)):
                values[i] = int(values[i])
    return values


def parse_ints(numbers):
    return [int(num) for num in parse_numbers(numbers)]


def parse_dates(numbers):
    return parse_numbers(numbers, True)


def parse_colors(colors):
    """
    """
    firstList = colors.split(",")
    numList = []
    finalList = []

    for string in firstList:
        if "[" in string:   # for rgba args
            if not numList:
                string = string.replace("[", "")
                numList.append(float(string))
            else:
                verif.util.error("Invalid rgba arg \"{}\"".format(string))

        elif "]" in string:
            if numList:
                string = string.replace("]", "")
                numList.append(float(string))
                finalList.append(numList)
                numList = []
            else:
                verif.util.error("Invalid rgba arg \"{}\"".format(string))

        # append to rgba lists if present, otherwise grayscale intensity
        elif verif.util.is_number(string):
            if numList:
                numList.append(float(string))
            else:
                finalList.append(string)

        else:
            if not numList:  # string args and hexcodes
                finalList.append(string)
            else:
                verif.util.error("Cannot read color args.")
    return finalList


def subplot(i, N):
    """ Sets up subplot for index i (starts at 0) out of N """
    [X, Y] = get_subplot_size(N)
    mpl.subplot(Y, X, i + 1)


def get_subplot_size(N):
    Y = 1
    if N > 4:
        Y = np.ceil(np.sqrt(N) / 1.5)
    X = np.ceil(N / Y)
    return [int(X), int(Y)]


def get_map_resolution(lats, lons):
    dlat = (max(lats) - min(lats))
    dlon = (max(lons) - min(lons))
    scale = max(dlat, dlon)
    if np.isnan(scale):
        res = "c"
    elif scale > 60:
        res = "c"
    elif scale > 1:
        res = "i"
    elif scale > 0.001:
        res = "h"
    elif scale > 0.0001:
        res = "f"
    else:
        res = "c"
    return res


def get_cartopy_map_resolution(lats, lons):
    dlat = (max(lats) - min(lats))
    dlon = (max(lons) - min(lons))
    scale = max(dlat, dlon)
    return 5
    # TODO: Implement this
    if np.isnan(scale):
        res = "c"
    elif scale > 60:
        res = "c"
    elif scale > 1:
        res = "i"
    elif scale > 0.001:
        res = "h"
    elif scale > 0.0001:
        res = "f"
    else:
        res = "c"
    return res


def fill(x, y_lower, y_upper, col, alpha=1, zorder=0, hatch=''):
    """ Fill an area between two curves

    Fill an area along x, between y_lower and y_upper. Both y_lower and y_upper most
    correspond to points in x (i.e. be in the same order)

    Arguments:
       x (np.array): x-axis values
       y_lower (np.array): y-axis values for lower envelope
       y_upper (np.array): y-axis values for upper envelope
       col: Color of filled area in any format understood by mpl.fill
       alpha: alpha of filled area
       zorder: zorder
       hatch: any hatch string understood by mpl.fill
    """
    # Populate a list of non-missing points
    X = list()
    Y = list()
    for i in range(0, len(x)):
        if not(np.isnan(x[i]) or np.isnan(y_lower[i])):
            X.append(x[i])
            Y.append(y_lower[i])
    for i in range(len(x) - 1, -1, -1):
        if not (np.isnan(x[i]) or np.isnan(y_upper[i])):
            X.append(x[i])
            Y.append(y_upper[i])
    if len(X) > 0:
        mpl.fill(X, Y, facecolor=col, alpha=alpha, linewidth=0, zorder=zorder,
              hatch=hatch)


def clean(data):
    """ Copy and sanitize data from a netCDF4 variable

    Arguments:
       data: A netCDF4 variable

    Returns:
       np.array: A numpy array where invalid values have been set to np.nan

    """
    if len(data.shape) == 1 and data.shape[0] == 0:
        return np.zeros(0)

    data = data[:].astype(float)
    q = np.ma.filled(data, fill_value=-999)
    # Remove missing values. Convert to -999 and then back to nan to avoid
    # warning messages when doing <, >, and == comparisons with nan.
    q[np.isnan(q)] = -999
    q[(q == -999) | (q > 1e30)] = np.nan
    return q


def get_date(date, diff):
    """ Date calculation: Adds 'diff' to 'date'

    Arguments:
       date (int): An integer of the form YYYYMMDD
       diff (int): Number of days to add to date

    Returns:
       int: A new date in the form YYYYMMDD
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
    if len(I) > 0:
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
    """ Use bin_type to turn array into binary values """
    I = np.isnan(array) == 0
    array = copy.deepcopy(array)
    if bin_type == "below":
        array[I] = array[I] < threshold
    elif bin_type == "below=":
        array[I] = array[I] <= threshold
    elif bin_type == "above":
        array[I] = array[I] > threshold
    elif bin_type == "above=":
        array[I] = array[I] >= threshold
    elif upper_threshold is None:
        error("Cannot apply thresholding with bin_type '%s'" % bin_type)
    elif bin_type == "within":
        array[I] = (array[I] > threshold) & (array[I] < upper_threshold)
    elif bin_type == "within=":
        array[I] = (array[I] > threshold) & (array[I] <= upper_threshold)
    elif bin_type == "=within":
        array[I] = (array[I] >= threshold) & (array[I] < upper_threshold)
    elif bin_type == "=within=":
        array[I] = (array[I] >= threshold) & (array[I] <= upper_threshold)
    array = np.array(array, float)
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
        lower_eq = False
        upper_eq = False

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
            upper_eq = True
        if bin_type in ["above=", "=within", "=within="]:
            lower_eq = True
        intervals.append(verif.interval.Interval(lower, upper, lower_eq, upper_eq))
    return intervals


def almost_equal(value1, value2, tol=1e-7):
    return abs(value1 - value2) < tol


def is_valid_nc(filename):
    """ Return True if the file is a valid NetCDF file """
    try:
        file = netcdf(filename, 'r')
        file.close()
        return True
    except Exception:
        return False


def get_distance_matrix(locations):
    """ Computes distance matrix betweeen locations

    Arguments:
       locations (list): List of verif.location.Location)

    Returns:
       np.array: 2D array with the distance between each pair of locations in meters
    """

    N = len(locations)
    dist = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            dist[i, j] = locations[i].get_distance(locations[j])
    return dist


def proj4_string_to_dict(string):
    """ Parse a proj4 string and create a dictionary

    Arguments:
       string (str): A proj4 string like "+proj=lcc +lat_0=63
       +lon_0=15 +lat_1=63 +lat_2=63 +no_defs +R=6.371e+06"
    Returns:
       dict: A dictionary of attributes and values, e.g =proj=
    """
    r = dict()
    pairs = string.split(' ')
    for pair in pairs:
        keyvalue = pair.split('=')
        if len(keyvalue) > 2:
            verif.util.error("Could not parse proj4 parameter: %s" % pair)
        key = keyvalue[0]
        if len(keyvalue) == 1:
            value = True
        else:
            value = keyvalue[1]
        # Try converting to number
        try:
            value = float(value)
        except Exception:
            pass
        r[key] = value
    return r


def parse_label(string):
    """
    Replaces newlines
    """
    return str(string).replace('\\n', '\n')
