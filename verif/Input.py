from scipy import io
import numpy as np
from Station import *
import Common

# Abstract base class representing verification data
class Input:
   def __init__(self, filename):
      self._filename = filename
   def getName(self):
      pass
   def getFilename(self):
      return self.getName()
   def getDates(self):
      pass
   def getStations(self):
      pass
   def getOffsets(self):
      pass
   def getThresholds(self):
      pass
   def getQuantiles(self):
      pass
   def getObs(self):
      pass
   def getDeterministic(self):
      pass
   def getOther(self, name):
      pass
   def getMetrics(self):
      pass
   def getFilename(self):
      return self._filename
   def getVariables(self):
      pass
   def getUnits(self):
      pass
   def getLats(self):
      stations = self.getStations()
      lats = np.zeros(len(stations))
      for i in range(0, len(stations)):
         lats[i] = stations[i].lat()
      return lats
   def getLons(self):
      stations = self.getStations()
      lons = np.zeros(len(stations))
      for i in range(0, len(stations)):
         lons[i] = stations[i].lon()
      return lons
   def getStationIds(self):
      stations = self.getStations()
      ids = np.zeros(len(stations))
      for i in range(0, len(stations)):
         ids[i] = stations[i].id()
      return ids

# Original fileformat used by OutputVerif in COMPS
class Comps(Input):
   def __init__(self, filename):
      Input.__init__(self, filename)
      self._file = io.netcdf.netcdf_file(filename, 'r')
   def getName(self):
      return self._file.variables
   def getStations(self):
      lat  = Common.clean(self._file.variables["Lat"])
      lon  = Common.clean(self._file.variables["Lon"])
      id   = Common.clean(self._file.variables["Location"])
      elev = Common.clean(self._file.variables["Elev"])
      stations = list()
      for i in range(0, lat.shape[0]):
         station = Station(id, lat, lon, elev)
         stations.append(station)
      return stations
   def getScores(self, metric):
      temp = Common.clean(self._file.variables[metric])
      return temp
   def getDates(self):
      return Common.clean(self._file.variables["Date"])
   def getOffsets(self):
      return Common.clean(self._file.variables["Offset"])
   def getMetrics(self):
      metrics = list()
      for (metric, v) in self._file.variables.iteritems():
         if(not metric in ["Date", "Offset", "Location", "Lat", "Lon", "Elev"]):
            metrics.append(metric)
      return metrics
   def getVariables(self):
      metrics = list()
      for (metric, v) in self._file.variables.iteritems():
         metrics.append(metric)
      return metrics
   def getUnits(self):
      if(hasattr(self._file, "Units")):
         if(self._file.Units == ""):
            return "No units"
         elif(self._file.Units == "%"):
            return "%"
         else:
            return "$" + self._file.Units + "$"
      else:
         return "No units"
   def getVariable(self):
      return self._file.Variable

# New standard format, based on NetCDF/CF
class NetcdfCf(Input):
   def __init__(self, filename):
      pass

class Fake(Input):
   def __init__(self, obs, fcst):
      self._obs = obs
      self._fcst = fcst
   def getObs(self):
      return self._obs
   def getMean(self):
      return self._fcst
