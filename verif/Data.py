from scipy import io
import numpy as np
import verif.Common as Common
import re
import sys
import os
import verif.Input as Input
from matplotlib.dates  import *
from matplotlib.ticker import ScalarFormatter

# Access verification data from a set of COMPS NetCDF files
# Only returns data that is available for all files, for fair comparisons
# i.e if some dates/offsets/locations are missing
#
# filenames: COMPS NetCDF verification files
# dates: Only allow these dates
# offsets: Only allow these offsets
# locations: Only allow these locationIds
# clim: Use this NetCDF file to compute anomaly. Should therefore be a climatological
#       forecast. Subtract/divide the forecasts from this file from all forecasts and
#       observations from the other files.
# climType: 'subtract', or 'divide' the climatology
# training: Remove the first 'training' days of data (to allow the forecasts to train its
#           adaptive parameters)
class Data:
   def __init__(self, filenames, dates=None, offsets=None, locations=None, latlonRange=None, elevRange=None, clim=None,
         climType="subtract", training=None):
      if(not isinstance(filenames, list)):
         filenames = [filenames]
      self._axis = "date"
      self._index = 0

      # Organize files
      self._files = list()
      self._cache  = list()
      self._clim = None
      for filename in filenames:
         if(not os.path.exists(filename)):
            Common.error("File '" + filename + "' is not a valid input file")
         # file = io.netcdf.netcdf_file(filename, 'r')
         try:
            file = Input.Comps(filename)
         except:
            file = Input.Text(filename)
         self._files.append(file)
         self._cache.append(dict())
      if(clim != None):
         self._clim = io.netcdf.netcdf_file(clim, 'r')
         self._cache.append(dict())
         if(not (climType == "subtract" or climType == "divide")):
            Common.error("Data: climType must be 'subtract' or 'divide")
         self._climType = climType

      # Climatology file
         self._files = self._files + [self._clim]

      # Latitude-Longitude range
      if(latlonRange != None):
         lat   = self._files[0].getLats()
         lon   = self._files[0].getLons()
         locId = self._files[0].getStationIds()
         latlonLocations = list()
         minLon = latlonRange[0]
         maxLon = latlonRange[1]
         minLat = latlonRange[2]
         maxLat = latlonRange[3]
         for i in range(0,len(lat)):
            currLat = float(lat[i])
            currLon = float(lon[i])
            if(currLat >= minLat and currLat <= maxLat and currLon >= minLon and currLon <= maxLon):
               latlonLocations.append(locId[i])
         useLocations = list()
         if(locations != None):
            for i in range(0, len(locations)):
               currLocation = locations[i]
               if(currLocation in latlonLocations):
                  useLocations.append(currLocation)
         else:
            useLocations = latlonLocations
         if(len(useLocations) == 0):
            Common.error("No available locations within lat/lon range")
      else:
         useLocations = locations

      # Elevation range
      if(elevRange != None):
         lat   = self._files[0].getElevs()
         locId = self._files[0].getStationIds()
         elevLocations = list()
         minElev = elevRange[0]
         maxElev = elevRange[1]
         for i in range(0,len(elev)):
            currElev = float(elev[i])
            if(currElev >= minElev and currElev <= maxElev):
               elevLocations.append(locId[i])
         useLocations = Common.intersect(useLocations, elevLocations)
         if(len(useLocations) == 0):
            Common.error("No available locations within elevation range")

      # Find common indicies
      self._datesI     = Data._getCommonIndices(self._files, "Date", dates)
      self._offsetsI   = Data._getCommonIndices(self._files, "Offset", offsets)
      self._locationsI = Data._getCommonIndices(self._files, "Location", useLocations)
      if(len(self._datesI[0]) == 0):
         Common.error("No valid dates selected")
      if(len(self._offsetsI[0]) == 0):
         Common.error("No valid offsets selected")
      if(len(self._locationsI[0]) == 0):
         Common.error("No valid locations selected")

      # Training
      if(training != None):
         for f in range(0, len(self._datesI)):
            if(len(self._datesI[f]) <= training):
               Common.error("Training period too long for " + self.getFilenames()[f] + \
                     ". Max training period is " + str(len(self._datesI[f])-1) + ".")
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
      doClim = self._clim != None and ("obs" in metrics or "fcst" in metrics)
      if(doClim):
         temp = self._getScore("fcst", len(self._files)-1)
         if(self._axis == "date"):
            clim = temp[self._index,:,:].flatten()
         elif(self._axis == "offset"):
            clim = temp[:,self._index,:].flatten()
         elif(self.isLocationAxis(self._axis)):
            clim = temp[:,:,self._index].flatten()
         elif(self._axis == "none" or self._axis == "threshold"):
            clim = temp.flatten()
         elif(self._axis == "all"):
            clim = temp
      else:
         clim = 0

      for i in range(0, len(metrics)):
         metric = metrics[i]
         temp = self._getScore(metric)
         #print self._axis

         if(self._axis == "date"):
            data[metric] = temp[self._index,:,:].flatten()
         elif(self._axis == "offset"):
            data[metric] = temp[:,self._index,:].flatten()
         elif(self.isLocationAxis(self._axis)):
            data[metric] = temp[:,:,self._index].flatten()
         elif(self._axis == "none" or self._axis == "threshold"):
            data[metric] = temp.flatten()
         elif(self._axis == "all"):
            data[metric] = temp
         else:
            Common.error("Data.py: unrecognized value of self._axis: " + self._axis)

         # Subtract climatology
         if(doClim and (metric == "fcst" or metric == "obs")):
            if(self._climType == "subtract"):
               data[metric] = data[metric] - clim
            else:
               data[metric] = data[metric] / clim

         # Remove missing values
         if(self._axis != "all"):
            currValid = (np.isnan(data[metric]) == 0) & (np.isinf(data[metric]) == 0)
            if(valid == None):
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
            q[i] = np.nan*np.zeros([1], 'float')

      return q

   # Find indicies of elements that are present in all files
   # Merge in values in 'aux' as well
   @staticmethod
   def _getCommonIndices(files, name, aux=None):
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
            for i in range(0,len(stations)):
               temp[i] = stations[i].id()
         if(values == None):
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
            for i in range(0,len(stations)):
               temp[i] = stations[i].id()
         I = np.where(np.in1d(temp, values))[0]
         II = np.zeros(len(I), 'int')
         for i in range(0,len(I)):
            II[i] = np.where(values[i] == temp)[0]

         indices.append(II)
      return indices

   def _getFiles(self):
      if(self._clim == None):
         return self._files
      else:
         return self._files[0:-1]

   def getMetrics(self):
      metrics = None
      for file in self._files:
         currMetrics = file.getMetrics()
         if(metrics == None):
            metrics = currMetrics
         else:
            metrics = set(metrics) & set(currMetrics)

      return metrics

   def _getIndices(self, axis, findex=None):
      if(axis == "date"):
         I = self._getDateIndices(findex)
      elif(axis == "offset"):
         I = self._getOffsetIndices(findex)
      elif(axis == "location"):
         I = self._getLocationIndices(findex)
      else:
         Common.error(axis)
      return I
   def _getDateIndices(self, findex=None):
      if(findex == None):
         findex = self._findex
      return self._datesI[findex]

   def _getOffsetIndices(self, findex=None):
      if(findex == None):
         findex = self._findex
      return self._offsetsI[findex]

   def _getLocationIndices(self, findex=None):
      if(findex == None):
         findex = self._findex
      return self._locationsI[findex]

   def _getScore(self, metric, findex=None):
      if(findex == None):
         findex = self._findex

      if(metric in self._cache[findex]):
         return self._cache[findex][metric]

      # Load all files
      for f in range(0, self.getNumFilesWithClim()):
         if(not metric in self._cache[f]):
            file = self._files[f]
            if(not metric in file.getVariables()):
               Common.error("Variable '" + metric + "' does not exist in " +
                     self.getFilenames()[f])
            temp = file.getScores(metric)
            dims = file.getDims(metric)
            temp = Common.clean(temp)
            for i in range(0, len(dims)):
               I = self._getIndices(dims[i].lower(), f)
               if(i == 0):
                  temp = temp[I,Ellipsis]
               if(i == 1):
                  temp = temp[:,I,Ellipsis]
               if(i == 2):
                  temp = temp[:,:,I,Ellipsis]
            self._cache[f][metric] = temp

      # Remove missing
      # If one configuration has a missing value, set all configurations to missing
      # This can happen when the dates are available, but have missing values
      isMissing = np.isnan(self._cache[0][metric])
      for f in range(1, self.getNumFilesWithClim()):
         isMissing = isMissing | (np.isnan(self._cache[f][metric]))
      for f in range(0, self.getNumFilesWithClim()):
         self._cache[f][metric][isMissing] = np.nan

      return self._cache[findex][metric]

   # Checks that all files have the variable
   def hasMetric(self, metric):
      for f in range(0, self.getNumFilesWithClim()):
         if(not metric in self._files[f].getVariables()):
            return False
      return True

   def setAxis(self, axis):
      self._index = 0 # Reset index
      self._axis = axis
   def setIndex(self, index):
      self._index = index
   def setFileIndex(self, index):
      self._findex = index
   def getNumFiles(self):
      return len(self._files) - (self._clim != None)
   def getNumFilesWithClim(self):
      return len(self._files)

   def getUnits(self):
      # TODO: Only check first file?
      return self._files[0].getUnits()

   def isLocationAxis(self, axis):
      if(axis == None):
         return False
      prog = re.compile("location.*")
      return prog.match(axis)

   def getAxisSize(self, axis=None):
      if(axis == None):
         axis = self._axis
      return len(self.getAxisValues(axis))

   # What values represent this axis?
   def getAxisValues(self, axis=None):
      if(axis == None):
         axis = self._axis
      if(axis == "date"):
         return Common.convertDates(self._getScore("Date").astype(int))
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
            Common.error("Data.getAxisValues has a bad axis name: " + axis)
         return data
      else:
         return [0]
   def isAxisContinuous(self, axis=None):
      if(axis == None):
         axis = self._axis
      return axis in ["date", "offset", "threshold"]

   def getAxisFormatter(self, axis=None):
      if(axis == None):
         axis = self._axis
      if(axis == "date"):
         return DateFormatter('\n%Y-%m-%d')
      else:
         return ScalarFormatter()


   # filename including path
   def getFullFilenames(self):
      names = list()
      files = self._getFiles()
      for i in range(0, len(files)):
         names.append(files[i].getFilename())
      return names
   def getFilename(self, findex=None):
      if(findex == None):
         findex = self._findex
      return self.getFilenames()[findex]

   # Do not include the path
   def getFilenames(self):
      names = self.getFullFilenames()
      for i in range(0, len(names)):
         I = names[i].rfind('/')
         names[i] = names[i][I+1:]
      return names
   def getShortNames(self):
      names = self.getFilenames()
      for i in range(0, len(names)):
         I = names[i].rfind('.')
         names[i] = names[i][:I]
      return names

   def getAxis(self, axis=None):
      if(axis == None):
         axis = self._axis
      return axis

   def getVariable(self):
      return self._files[0].getVariable()

   def getVariableAndUnits(self):
      return self.getVariable() + " (" + self.getUnits() + ")"

   def getX0(self):
      x0 = None
      prog = re.compile("Precip.*")
      if(prog.match(self.getVariable())):
         x0 = 0
      return x0

   def getX1(self):
      x1 = None
      prog = re.compile("RH")
      if(prog.match(self.getVariable())):
         x1 = 100
      return x1

   def getAxisLabel(self, axis=None):
      if(axis == None):
         axis = self._axis
      if(axis == "date"):
         return "Date"
      elif(axis == "offset"):
         return "Offset (h)"
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
      if(axis == None):
         axis = self._axis
      prog = re.compile("location.*")
      if(prog.match(axis)):
         descs = list()
         ids = self._getScore("Location")
         lats = self._getScore("Lat")
         lons = self._getScore("Lon")
         elevs = self._getScore("Elev")
         for i in range(0, len(ids)):
            string = "%6d %5.2f %5.2f %5.0f" % (ids[i],lats[i], lons[i], elevs[i])
            descs.append(string)
         return descs
      if(axis == "date"):
         values = self.getAxisValues(axis)
         values = num2date(values)
         dates = list()
         for i in range(0, len(values)):
            dates = dates + [values[i].strftime("%Y/%m/%d")]
         return dates
      else:
         return self.getAxisValues(axis)

   def getAxisDescriptionHeader(self, axis=None):
      if(axis == None):
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
      elif(axis == "location" or axis == "locationId" or axis == "locationElev" or axis == "locationLat" or axis == "locationLon"):
         return 2
      else:
         return None

   def getPvar(self, threshold):
      minus = ""
      if(threshold < 0):
         # Negative thresholds
         minus = "m"
      if(abs(threshold - int(threshold)) > 0.01):
         var = "p" + minus + str(abs(threshold)).replace(".", "")
      else:
         var   = "p" + minus + str(int(abs(threshold)))
      return var

   def getQvar(self, quantile):
      quantile = quantile * 100
      minus = ""
      if(abs(quantile - int(quantile)) > 0.01):
         var = "q" + minus + str(abs(quantile)).replace(".", "")
      else:
         var   = "q" + minus + str(int(abs(quantile)))

      if(not self.hasMetric(var) and quantile == 50):
         Common.warning("Could not find q50, using fcst instead")
         return "fcst"
      return var
