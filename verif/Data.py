from scipy import io
import numpy as np
import verif.Util as Util
import re
import sys
import os
import verif.Input as Input
from matplotlib.dates import *
import matplotlib.ticker


# Access verification data from a set of Input files
# Only returns data that is available for all files, for fair comparisons
# i.e if some dates/offsets/locations are missing
#
# filenames: Input files. The types are autodetected.
# dates: Only allow these dates
# offsets: Only allow these offsets
# locations: Only allow these locationIds
# clim: Use this NetCDF file to compute anomaly. Should therefore be a
#       climatological forecast. Subtract/divide the forecasts from this file
#       from all forecasts and observations from the other files.
# climType: 'subtract', or 'divide' the climatology
# training: Remove the first 'training' days of data (to allow the forecasts to
#       train its adaptive parameters)
class Data:
   def __init__(self, filenames, dates=None, offsets=None, locations=None,
         latlonRange=None, elevRange=None, clim=None, climType="subtract",
         training=None, legend=None, removeMissingAcrossAll=True):
      if(not isinstance(filenames, list)):
         filenames = [filenames]
      self._axis = "date"
      self._index = 0
      self._removeMissingAcrossAll = removeMissingAcrossAll

      if(legend is not None and len(filenames) is not len(legend)):
         Util.error("Need one legend entry for each filename")
      self._legend = legend

      # Organize files
      self._files = list()
      self._cache = list()
      self._clim = None
      for filename in filenames:
         if(not os.path.exists(filename)):
            Util.error("File '" + filename + "' does not exist")
         if(Input.NetcdfCf.isValid(filename)):
            file = Input.NetcdfCf(filename)
         elif(Input.Comps.isValid(filename)):
            file = Input.Comps(filename)
         elif(Input.Text.isValid(filename)):
            file = Input.Text(filename)
         else:
            Util.error("File '" + filename + "' is not a valid input file")
         self._files.append(file)
         self._cache.append(dict())
      if(clim is not None):
         if(not os.path.exists(clim)):
            Util.error("File '" + clim + "' does not exist")
         if(Input.NetcdfCf.isValid(clim)):
            self._clim = Input.NetcdfCf(clim)
         elif(Input.Comps.isValid(clim)):
            self._clim = Input.Comps(clim)
         elif(Input.Text.isValid(clim)):
            self._clim = Input.Text(clim)
         else:
            Util.error("File '" + clim + "' is not a valid climatology file")
         self._cache.append(dict())
         if(not (climType == "subtract" or climType == "divide")):
            Util.error("Data: climType must be 'subtract' or 'divide")
         self._climType = climType

      # Climatology file
         self._files = self._files + [self._clim]

      # Latitude-Longitude range
      if(latlonRange is not None):
         lat = self._files[0].getLats()
         lon = self._files[0].getLons()
         locId = self._files[0].getStationIds()
         latlonLocations = list()
         minLon = latlonRange[0]
         maxLon = latlonRange[1]
         minLat = latlonRange[2]
         maxLat = latlonRange[3]
         for i in range(0, len(lat)):
            currLat = float(lat[i])
            currLon = float(lon[i])
            if(currLat >= minLat and currLat <= maxLat and
                  currLon >= minLon and currLon <= maxLon):
               latlonLocations.append(locId[i])
         useLocations = list()
         if(locations is not None):
            for i in range(0, len(locations)):
               currLocation = locations[i]
               if(currLocation in latlonLocations):
                  useLocations.append(currLocation)
         else:
            useLocations = latlonLocations
         if(len(useLocations) == 0):
            Util.error("No available locations within lat/lon range")
      elif locations is not None:
         useLocations = locations
      else:
         useLocations = self._files[0].getStationIds()

      # Elevation range
      if(elevRange is not None):
         stations = self._files[0].getStations()
         minElev = elevRange[0]
         maxElev = elevRange[1]
         elevLocations = list()
         for i in range(0, len(stations)):
            currElev = float(stations[i].elev())
            id = stations[i].id()
            if(currElev >= minElev and currElev <= maxElev):
               elevLocations.append(id)
         useLocations = Util.intersect(useLocations, elevLocations)
         if(len(useLocations) == 0):
            Util.error("No available locations within elevation range")

      # Find common indicies
      self._datesI = Data._getUtilIndices(self._files, "Date", dates)
      self._offsetsI = Data._getUtilIndices(self._files, "Offset", offsets)
      self._locationsI = Data._getUtilIndices(self._files, "Location",
            useLocations)
      if(len(self._datesI[0]) == 0):
         Util.error("No valid dates selected")
      if(len(self._offsetsI[0]) == 0):
         Util.error("No valid offsets selected")
      if(len(self._locationsI[0]) == 0):
         Util.error("No valid locations selected")

      # Training
      if(training is not None):
         for f in range(0, len(self._datesI)):
            if(len(self._datesI[f]) <= training):
               Util.error("Training period too long for " +
                     self.getFilenames()[f] + ". Max training period is " +
                     str(len(self._datesI[f]) - 1) + ".")
            self._datesI[f] = self._datesI[f][training:]

      self._findex = 0

   # Returns flattened arrays along the set axis/index
   def getScores(self, metrics):
      if(not isinstance(metrics, list)):
         metrics = [metrics]
      data = dict()
      valid = None
      axis = self._getAxisIndex(self._axis)

      # Compute climatology, if needed
      obsFcstAvailable = ("obs" in metrics or "fcst" in metrics)
      doClim = self._clim is not None and obsFcstAvailable
      if(doClim):
         temp = self._getScore("fcst", len(self._files) - 1)
         if(self._axis == "date"):
            clim = temp[self._index, :, :].flatten()
         elif(self._axis == "month"):
            dates = self.getAxisValues("date")
            months = self.getAxisValues("month")
            if(self._index == months.shape[0]-1):
               I = np.where(dates >= months[self._index])
            else:
               I = np.where((dates >= months[self._index]) &
                            (dates < months[self._index+1]))
            clim = temp[I, :, :].flatten()
         elif(self._axis == "year"):
            dates = self.getAxisValues("date")
            years = self.getAxisValues("year")
            if(self._index == years.shape[0]-1):
               I = np.where(dates >= years[self._index])
            else:
               I = np.where((dates >= years[self._index]) &
                            (dates < years[self._index+1]))
            clim = temp[I, :, :].flatten()
         elif(self._axis == "offset"):
            clim = temp[:, self._index, :].flatten()
         elif(self.isLocationAxis(self._axis)):
            clim = temp[:, :, self._index].flatten()
         elif(self._axis == "none" or self._axis == "threshold"):
            clim = temp.flatten()
         elif(self._axis == "all"):
            clim = temp
      else:
         clim = 0

      for i in range(0, len(metrics)):
         metric = metrics[i]
         temp = self._getScore(metric)
         # print self._axis

         if(self._axis == "date"):
            data[metric] = temp[self._index, :, :].flatten()
         elif(self._axis == "month"):
            dates = self.getAxisValues("date")
            months = self.getAxisValues("month")
            if(self._index == months.shape[0]-1):
               I = np.where(dates >= months[self._index])
            else:
               I = np.where((dates >= months[self._index]) &
                            (dates < months[self._index+1]))
            data[metric] = temp[I, :, :].flatten()
         elif(self._axis == "year"):
            dates = self.getAxisValues("date")
            years = self.getAxisValues("year")
            if(self._index == years.shape[0]-1):
               I = np.where(dates >= years[self._index])
            else:
               I = np.where((dates >= years[self._index]) &
                            (dates < years[self._index+1]))
            data[metric] = temp[I, :, :].flatten()
         elif(self._axis == "offset"):
            data[metric] = temp[:, self._index, :].flatten()
         elif(self.isLocationAxis(self._axis)):
            data[metric] = temp[:, :, self._index].flatten()
         elif(self._axis == "none" or self._axis == "threshold"):
            data[metric] = temp.flatten()
         elif(self._axis == "all"):
            data[metric] = temp
         else:
            Util.error("Data.py: unrecognized value of self._axis: " +
                  self._axis)

         # Subtract climatology
         if(doClim and (metric == "fcst" or metric == "obs")):
            if(self._climType == "subtract"):
               data[metric] = data[metric] - clim
            else:
               data[metric] = data[metric] / clim

         # Remove missing values
         if(self._axis != "all"):
            currValid = (np.isnan(data[metric]) == 0)\
                      & (np.isinf(data[metric]) == 0)
            if(valid is None):
               valid = currValid
            else:
               valid = (valid & currValid)
      if(self._axis != "all"):
         I = np.where(valid)

      q = list()
      for i in range(0, len(metrics)):
         if(self._axis != "all"):
            q.append(data[metrics[i]][I])
         else:
            q.append(data[metrics[i]])

      # No valid data
      if(q[0].shape[0] == 0):
         for i in range(0, len(metrics)):
            q[i] = np.nan * np.zeros([1], 'float')

      return q

   def getStations(self):
      stations = self._files[0].getStations()
      I = self._locationsI[0]
      useStations = list()
      for i in I:
         useStations.append(stations[i])
      return useStations

   # Find indicies of elements that are present in all files
   # Merge in values in 'aux' as well
   @staticmethod
   def _getUtilIndices(files, name, aux=None):
      # Find common values among all files
      values = aux
      for file in files:
         if(name == "Date"):
            temp = file.getDates()
         elif(name == "Offset"):
            temp = file.getOffsets()
         elif(name == "Location"):
            stations = file.getStations()
            temp = np.zeros(len(stations))
            for i in range(0, len(stations)):
               temp[i] = stations[i].id()
         if(values is None):
            values = temp
         else:
            values = np.intersect1d(values, temp)
      # Sort values, since for example, dates may not be in an ascending order
      values = np.sort(values)

      # Determine which index each value is at
      indices = list()
      for file in files:
         if(name == "Date"):
            temp = file.getDates()
         elif(name == "Offset"):
            temp = file.getOffsets()
         elif(name == "Location"):
            stations = file.getStations()
            temp = np.zeros(len(stations))
            for i in range(0, len(stations)):
               temp[i] = stations[i].id()
         I = np.where(np.in1d(temp, values))[0]
         II = np.zeros(len(I), 'int')
         for i in range(0, len(I)):
            II[i] = np.where(values[i] == temp)[0]

         indices.append(II)
      return indices

   def _getFiles(self):
      if(self._clim is None):
         return self._files
      else:
         return self._files[0:-1]

   def getMetrics(self):
      metrics = None
      for file in self._files:
         currMetrics = file.getMetrics()
         if(metrics is None):
            metrics = currMetrics
         else:
            metrics = set(metrics) & set(currMetrics)

      return metrics

   def getThresholds(self):
      thresholds = None
      for file in self._files:
         currThresholds = file.getThresholds()
         if(thresholds is None):
            thresholds = currThresholds
         else:
            thresholds = set(thresholds) & set(currThresholds)

      thresholds = sorted(thresholds)
      return thresholds

   def getQuantiles(self):
      quantiles = None
      for file in self._files:
         currQuantiles = file.getQuantiles()
         if(quantiles is None):
            quantiles = currQuantiles
         else:
            quantiles = set(quantiles) & set(currQuantiles)

      quantiles = sorted(quantiles)
      return quantiles

   # Get the names of all quantile scores
   def getQuantileNames(self):
      quantiles = self.getQuantiles()
      return [self.getQvar(quantile / 100) for quantile in quantiles]

   def _getIndices(self, axis, findex=None):
      if(axis == "date"):
         I = self._getDateIndices(findex)
      elif(axis == "offset"):
         I = self._getOffsetIndices(findex)
      elif(axis == "location"):
         I = self._getLocationIndices(findex)
      else:
         Util.error("Could not get indices for axis: " + str(axis))
      return I

   def _getDateIndices(self, findex=None):
      if(findex is None):
         findex = self._findex
      return self._datesI[findex]

   def _getOffsetIndices(self, findex=None):
      if(findex is None):
         findex = self._findex
      return self._offsetsI[findex]

   def _getLocationIndices(self, findex=None):
      if(findex is None):
         findex = self._findex
      return self._locationsI[findex]

   def _getScore(self, metric, findex=None):
      if(findex is None):
         findex = self._findex

      if(metric in self._cache[findex]):
         return self._cache[findex][metric]

      # Load all files
      for f in range(0, self.getNumFilesWithClim()):
         if(metric not in self._cache[f]):
            file = self._files[f]
            if(metric not in file.getVariables()):
               Util.error("Variable '" + metric + "' does not exist in " +
                     self.getFilenames()[f])
            temp = file.getScores(metric)
            dims = file.getDims(metric)
            temp = Util.clean(temp)
            for i in range(0, len(dims)):
               I = self._getIndices(dims[i].lower(), f)
               if(i == 0):
                  temp = temp[I, Ellipsis]
               if(i == 1):
                  temp = temp[:, I, Ellipsis]
               if(i == 2):
                  temp = temp[:, :, I, Ellipsis]
            self._cache[f][metric] = temp

      # Remove missing. If one configuration has a missing value, set all
      # configurations to missing This can happen when the dates are available,
      # but have missing values
      if self._removeMissingAcrossAll:
         isMissing = np.isnan(self._cache[0][metric])
         for f in range(1, self.getNumFilesWithClim()):
            isMissing = isMissing | (np.isnan(self._cache[f][metric]))
         for f in range(0, self.getNumFilesWithClim()):
            self._cache[f][metric][isMissing] = np.nan

      return self._cache[findex][metric]

   # Checks that all files have the variable
   def hasMetric(self, metric):
      for f in range(0, self.getNumFilesWithClim()):
         if(metric not in self._files[f].getVariables()):
            return False
      return True

   # Set the axis along which data is aggregated. One of offset, date, month,
   # year, location, locationLat, locationLon, locationElev.
   def setAxis(self, axis):
      self._index = 0  # Reset index
      self._axis = axis

   def setIndex(self, index):
      self._index = index

   def setFileIndex(self, index):
      self._findex = index

   def getNumFiles(self):
      return len(self._files) - (self._clim is not None)

   def getNumFilesWithClim(self):
      return len(self._files)

   def getUnits(self):
      # TODO: Only check first file?
      return self._files[0].getUnits()

   def isLocationAxis(self, axis):
      if(axis is None):
         return False
      prog = re.compile("location.*")
      return prog.match(axis)

   def getAxisSize(self, axis=None):
      if(axis is None):
         axis = self._axis
      return len(self.getAxisValues(axis))

   # What values represent this axis?
   def getAxisValues(self, axis=None):
      if(axis is None):
         axis = self._axis
      if(axis == "date"):
         return Util.convertDates(self._getScore("Date").astype(int))
      elif(axis == "month"):
         dates = self._getScore("Date").astype(int)
         months = np.unique((dates / 100) * 100 + 1)
         return Util.convertDates(months)
      elif(axis == "year"):
         dates = self._getScore("Date").astype(int)
         years = np.unique((dates / 10000) * 10000 + 101)
         return Util.convertDates(years)
      elif(axis == "offset"):
         return self._getScore("Offset").astype(int)
      elif(axis == "none"):
         return [0]
      elif(self.isLocationAxis(axis)):
         if(axis == "location"):
            data = range(0, len(self._getScore("Location")))
         elif(axis == "locationId"):
            data = self._getScore("Location").astype(int)
         elif(axis == "locationElev"):
            data = self._getScore("Elev")
         elif(axis == "locationLat"):
            data = self._getScore("Lat")
         elif(axis == "locationLon"):
            data = self._getScore("Lon")
         else:
            Util.error("Data.getAxisValues has a bad axis name: " + axis)
         return data
      else:
         return [0]

   def isAxisContinuous(self, axis=None):
      if(axis is None):
         axis = self._axis
      return axis in ["date", "offset", "threshold", "month", "year"]

   def isAxisDate(self, axis=None):
      if(axis is None):
         axis = self._axis
      return axis in ["date", "month", "year"]

   def getAxisFormatter(self, axis=None):
      if(axis is None):
         axis = self._axis
      if(axis == "date"):
         return DateFormatter('\n%Y-%m-%d')
      elif(axis == "month"):
         return DateFormatter('\n%Y-%m')
      elif(axis == "year"):
         return DateFormatter('\n%Y')
      else:
         return matplotlib.ticker.ScalarFormatter()

   def getAxisLocator(self, axis=None):
      if(axis is None):
         axis = self._axis
      if(axis == "offset"):
         # Define our own locators, since in general we want multiples of 24
         # (or even fractions thereof) to make the ticks repeat each day. Aim
         # for a maximum of 12 ticks.
         offsets = self.getAxisValues("offset")
         span = max(offsets) - min(offsets)
         if(span > 300):
            return matplotlib.ticker.AutoLocator()
         elif(span > 200):
            return matplotlib.ticker.MultipleLocator(48)
         elif(span > 144):
            return matplotlib.ticker.MultipleLocator(24)
         elif(span > 72):
            return matplotlib.ticker.MultipleLocator(12)
         elif(span > 36):
            return matplotlib.ticker.MultipleLocator(6)
         elif(span > 12):
            return matplotlib.ticker.MultipleLocator(3)
         else:
            return matplotlib.ticker.MultipleLocator(1)
      else:
         return matplotlib.ticker.AutoLocator()

   # filename including path
   def getFullFilenames(self):
      names = list()
      files = self._getFiles()
      for i in range(0, len(files)):
         names.append(files[i].getFilename())
      return names

   def getFilename(self, findex=None):
      if(findex is None):
         findex = self._findex
      return self.getFilenames()[findex]

   # Do not include the path
   def getFilenames(self):
      names = self.getFullFilenames()
      for i in range(0, len(names)):
         I = names[i].rfind('/')
         names[i] = names[i][I + 1:]
      return names

   def getLegend(self):
      if(self._legend is None):
         legend = self.getFilenames()
      else:
         legend = self._legend
      return legend

   def getShortNames(self):
      names = self.getFilenames()
      for i in range(0, len(names)):
         I = names[i].rfind('.')
         names[i] = names[i][:I]
      return names

   def getAxis(self, axis=None):
      if(axis is None):
         axis = self._axis
      return axis

   def getVariable(self):
      return self._files[0].getVariable()

   def getVariableAndUnits(self):
      var = self.getVariable()
      return var.name() + " (" + var.units() + ")"

   def getAxisLabel(self, axis=None):
      if(axis is None):
         axis = self._axis
      if(axis == "date"):
         return "Date"
      elif(axis == "offset"):
         return "Offset (h)"
      elif(axis == "month"):
         return "Month"
      elif(axis == "year"):
         return "Year"
      elif(axis == "locationElev"):
         return "Elevation (m)"
      elif(axis == "locationLat"):
         return "Latitude ($^o$)"
      elif(axis == "locationLon"):
         return "Longitude ($^o$)"
      elif(axis == "threshold"):
         return self.getVariableAndUnits()

   def getLats(self):
      return self._getScore("Lat")

   def getLons(self):
      return self._getScore("Lon")

   def getElevs(self):
      return self._getScore("Elev")

   def getLocationIds(self):
      return self._getScore("Location")

   def getAxisDescriptions(self, axis=None):
      if(axis is None):
         axis = self._axis
      prog = re.compile("location.*")
      if(prog.match(axis)):
         descs = list()
         ids = self._getScore("Location")
         lats = self._getScore("Lat")
         lons = self._getScore("Lon")
         elevs = self._getScore("Elev")
         for i in range(0, len(ids)):
            string = "%6d %5.2f %5.2f %5.0f" % (ids[i], lats[i], lons[i],
                  elevs[i])
            descs.append(string)
         return descs
      if(self.isAxisDate(axis)):
         values = self.getAxisValues(axis)
         values = num2date(values)
         dates = list()
         for i in range(0, len(values)):
            dates = dates + [values[i].strftime("%Y/%m/%d")]
         return dates
      else:
         return self.getAxisValues(axis)

   def getAxisDescriptionHeader(self, axis=None):
      if(axis is None):
         axis = self._axis
      prog = re.compile("location.*")
      if(prog.match(axis)):
         return "%6s %5s %5s %5s" % ("id", "lat", "lon", "elev")
      else:
         return axis

   def _getAxisIndex(self, axis):
      if(axis == "date"):
         return 0
      elif(axis == "offset"):
         return 1
      elif(axis == "location" or axis == "locationId" or axis == "locationElev" or
            axis == "locationLat" or axis == "locationLon"):
         return 2
      else:
         return None

   @staticmethod
   def getPvar(threshold):
      return "p%g" % (threshold)

   def getQvar(self, quantile):
      quantile = quantile * 100
      minus = ""
      if(abs(quantile - int(quantile)) > 0.01):
         var = "q" + minus + str(abs(quantile)).replace(".", "")
      else:
         var = "q" + minus + str(int(abs(quantile)))

      if(not self.hasMetric(var) and quantile == 50):
         Util.warning("Could not find q50, using fcst instead")
         return "fcst"
      return var
