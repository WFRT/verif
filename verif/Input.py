from scipy import io
import numpy as np
import verif.Station as Station
import verif.Common as Common

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
         station = Station.Station(id[i], lat[i], lon[i], elev[i])
         stations.append(station)
      return stations
   def getScores(self, metric):
      temp = Common.clean(self._file.variables[metric])
      return temp
   def getDims(self, metric):
      return self._file.variables[metric].dimensions
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

# Flat text file format
class Text(Input):
   def __init__(self, filename):
      import csv
      Input.__init__(self, filename)
      file = open(filename, 'r')
      reader = csv.reader(file, delimiter=' ')
      # First pass, get the dimensions
      self._dates = set()
      self._offsets = set()
      self._stations = set()
      obs = dict()
      fcst = dict()
      for row in reader:
         if(row[0] != '#'):
            date = float(row[0])
            offset = float(row[1])
            self._dates.add(date)
            self._offsets.add(offset)
            lat = float(row[2])
            lon = float(row[3])
            elev = float(row[4])
            station = Station.Station(0, lat, lon, elev)
            self._stations.add(station)
            print (date,offset,lat,lon,elev)
            obs[(date,offset,lat,lon,elev)]  = float(row[5])
            fcst[(date,offset,lat,lon,elev)] = float(row[6])
      file.close()
      self._dates = list(self._dates)
      self._offsets = list(self._offsets)
      self._stations = list(self._stations)
      Ndates = len(self._dates)
      Noffsets = len(self._offsets)
      Nlocations = len(self._stations)
      self._obs  = np.zeros([Ndates, Noffsets, Nlocations], 'float') * np.nan
      self._fcst = np.zeros([Ndates, Noffsets, Nlocations], 'float') * np.nan
      for d in range(0,len(self._dates)):
         date = self._dates[d]
         for o in range(0, len(self._offsets)):
            offset = self._offsets[o]
            for s in range(0, len(self._stations)):
               station = self._stations[s]
               lat = station.lat()
               lon = station.lon()
               elev = station.elev()
               key = (date,offset,lat,lon,elev)
               if(obs.has_key(key)):
                  self._obs[d][o][s]  = obs[key]
               if(fcst.has_key(key)):
                  self._fcst[d][o][s] = fcst[key]

      counter = 0
      for station in self._stations:
         station.id(counter)
         counter = counter + 1
      self._dates = np.array(self._dates)
      self._offsets = np.array(self._offsets)
   def getName(self):
      return "Unknown"
   def getStations(self):
      return self._stations
   def getScores(self, metric):
      if(metric == "obs"):
         return self._obs
      elif(metric == "fcst"):
         return self._fcst
      elif(metric == "Offset"):
         return self._offsets
      else:
         Common.error("Cannot find " + metric)
   def getDims(self, metric):
      if(metric == "obs" or metric == "fcst"):
         return ["Date", "Offset", "Location"]
      else:
         return [metric]
   def getDates(self):
      return self._dates
   def getOffsets(self):
      return self._offsets
   def getMetrics(self):
      metrics = ["obs", "fcst"]
      return metrics
   def getVariables(self):
      metrics = ["obs", "fcst", "Date", "Offset", "Location", "Lat", "Lon", "Elev"]
      return metrics
   def getUnits(self):
      return "Unknown units"
   def getVariable(self):
      return "Unknown"

class Fake(Input):
   def __init__(self, obs, fcst):
      self._obs = obs
      self._fcst = fcst
   def getObs(self):
      return self._obs
   def getMean(self):
      return self._fcst
