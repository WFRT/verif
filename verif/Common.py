import datetime
import numpy as np
import sys
from matplotlib.dates import *
from copy import deepcopy
import matplotlib.pyplot as mpl
def convertDates(dates):
   numDates = len(dates)
   dates2 = np.zeros([numDates], 'float')   
   for i in range(0, numDates):
      year = int(dates[i] / 10000)
      month = int(dates[i] / 100 % 100)
      day = int(dates[i] % 100)
      dates2[i] = date2num(datetime.datetime(year, month, day, 0))
   return dates2

def red(text):
   return "\033[31m"+text+"\033[0m"

def removeMargin():
   mpl.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0,hspace=0)

def green(text):
   return "\033[32m"+text+"\033[0m"

def yellow(text):
   return "\033[33m"+text+"\033[0m"

def experimental():
   return yellow("(experimental)")

def error(message):
   print "\033[1;31mError: " + message + "\033[0m"
   sys.exit(1)

def warning(message):
   print "\033[1;33mWarning: " + message + "\033[0m"

# allowable formats:
# num
# num1,num2,num3
# start:end
# start:step:end
def parseNumbers(numbers):
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
         step  = 1
         if(len(colonList) == 3):
            step = float(colonList[1])
         end   = float(colonList[-1]) + 0.0001 # arange does not include the end point
         values = values + list(np.arange(start, end, step))
      else:
         error("Could not translate '" + numbers + "' into numbers")
   return values

def testParseNumbers():
   print parseNumbers("1,2,3:5,6,7:2:20")
   print parseNumbers("1")

# Sets up subplot for index i (starts at 0) out of N
def subplot(i, N):
   [X,Y] = getSubplotSize(N)
   mpl.subplot(Y,X,i+1)

def getSubplotSize(N):
   Y = 1
   if(N > 4):
      Y= np.ceil(np.sqrt(N)/1.5)
   X = np.ceil(N / Y)
   return [int(X),int(Y)]
def getMapResolution(lats, lons):
   dlat = (max(lats) - min(lats))
   dlon = (max(lons) - min(lons))
   scale = max(dlat, dlon)
   if(np.isnan(scale)):
      res = "c"
   elif(scale > 60):
      res = "c"
   elif(scale > 1):
      res = "i"
   elif(scale > 0.1):
      res = "h"
   elif(scale > 0.01):
      res = "f"
   else:
      res = "c"
   return res

# Fill an area along x, between yLower and yUpper
# Both yLower and yUpper most correspond to points in x (i.e. be in the same order)
def fill(x, yLower, yUpper, col, alpha=1, zorder=0, hatch=''):
   # This approach doesn't work, because it doesn't remove points with missing x or y
   #X = np.hstack((x, x[::-1]))
   #Y = np.hstack((yLower, yUpper[::-1]))

   # Populate a list of non-missing points
   X = list()
   Y = list()
   for i in range(0,len(x)):
      if(not( np.isnan(x[i]) or np.isnan(yLower[i]))):
         X.append(x[i])
         Y.append(yLower[i])
   for i in range(len(x)-1, -1, -1):
      if(not (np.isnan(x[i]) or np.isnan(yUpper[i]))):
         X.append(x[i])
         Y.append(yUpper[i])
   if(len(X) > 0):
      mpl.fill(X, Y, facecolor=col, alpha=alpha,linewidth=0, zorder=zorder, hatch=hatch)

def clean(data):
   data = data[:].astype(float)
   q = deepcopy(data)
   mask = np.where(q == -999);
   q[mask] = np.nan
   mask = np.where(q < -100000);
   q[mask] = np.nan
   mask = np.where(q > 1e30);
   q[mask] = np.nan
   return q

# Date: YYYYMMDD diff: Add this many days
def getDate(date, diff):
   year  = int(date / 10000)
   month = int(date / 100 % 100)
   day   = int(date % 100)
   date2 = datetime.datetime(year, month, day, 0) + datetime.timedelta(diff)
   return int(date2.strftime('%Y%m%d'))

def nanmean(data, **args):
   return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).mean(**args),
         fill_value=np.nan)
def nanmedian(data, **args):
   I = np.where(np.isnan(data.flatten()) == 0)[0]
   return np.median(data.flatten()[I])
def nanmin(data, **args):
   return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).min(**args),
         fill_value=np.nan)
def nanmax(data, **args):
   return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).max(**args),
         fill_value=np.nan)
def nanstd(data, **args):
   return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).std(**args),
         fill_value=np.nan)
def nanpercentile(data, pers):
   I = np.where(np.isnan(data.flatten()) == 0)[0]
   p = np.percentile(data.flatten()[I], pers)
   return p
    #return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).percentile(pers),
    #      fill_value=np.nan)

def intersect(list1, list2):
   return list(set(list1) & set(list2))
