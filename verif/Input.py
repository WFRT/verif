from scipy import io
import numpy as np
import verif.Station as Station
import verif.Common as Common

# Abstract base class representing verification data
class Input:
   _description  = ""    # Overwrite this
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
   @classmethod
   def description(cls):
      return cls._description

# Original fileformat used by OutputVerif in COMPS
class Comps(Input):
   _description = "Netcdf format"
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
   @staticmethod
   def isValid(filename):
      try:
         file = io.netcdf.netcdf_file(filename, 'r')
      except:
         return False
      return True

# New standard format, based on NetCDF/CF
class NetcdfCf(Input):
   def __init__(self, filename):
      pass
   @staticmethod
   def isValid(filename):
      try:
         file = io.netcdf.netcdf_file(filename, 'r')
      except:
         return False
      valid = False
      if(hasattr(file, "Conventions")):
         if(file.Conventions == "verif_1.0.0"):
            valid = True
      file.close()
      return valid

# Flat text file format
class Text(Input):
   _description = "One line for each obs/fcst pair"
   def __init__(self, filename):
      import csv
      Input.__init__(self, filename)
      file = open(filename, 'r')
      reader = csv.reader(file, delimiter=' ')

      self._dates = set()
      self._offsets = set()
      self._stations = set()
      obs = dict()
      fcst = dict()
      indices = dict()
      header = None

      # Default values if columns not available
      offset  = 0
      date    = 0
      lat     = 0
      lon     = 0
      elev    = 0

      # Read the data into dictionary with (date,offset,lat,lon,elev) as key and obs/fcst as values
      for row in reader:
         if(header == None):
            # Parse the header so we know what each column represents
            header = row
            for i in range(0, len(header)):
               att = header[i]
               if(att == "date"):
                  indices["date"] = i
               elif(att == "offset"):
                  indices["offset"] = i
               elif(att == "lat"):
                  indices["lat"] = i
               elif(att == "lon"):
                  indices["lon"] = i
               elif(att == "elev"):
                  indices["elev"] = i
               elif(att == "obs"):
                  indices["obs"] = i
               elif(att == "fcst"):
                  indices["fcst"] = i

            # Ensure we have required columns
            requiredColumns = ["obs", "fcst"]
            for col in requiredColumns:
               if(not indices.has_key(col)):
                  msg = "Could not parse %s: Missing column '%s'" % (filename, col)
                  Common.error(msg)
         else:
            if(indices.has_key("date")):
               date = float(row[indices["date"]])
            self._dates.add(date)
            if(indices.has_key("offset")):
               offset = float(row[indices["offset"]])
            self._offsets.add(offset)
            if(indices.has_key("lat")):
               lat = float(row[indices["lat"]])
            if(indices.has_key("lon")):
               lon = float(row[indices["lon"]])
            if(indices.has_key("elev")):
               elev = float(row[indices["elev"]])
            station = Station.Station(0, lat, lon, elev)
            self._stations.add(station)
            obs[(date,offset,lat,lon,elev)]  = float(row[indices["obs"]])
            fcst[(date,offset,lat,lon,elev)] = float(row[indices["fcst"]])
      file.close()
      self._dates = list(self._dates)
      self._offsets = list(self._offsets)
      self._stations = list(self._stations)
      Ndates = len(self._dates)
      Noffsets = len(self._offsets)
      Nlocations = len(self._stations)

      # Put the dictionary data into a regular 3D array
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
   @staticmethod
   def isValid(filename):
      return True

class Fake(Input):
   def __init__(self, obs, fcst):
      self._obs = obs
      self._fcst = fcst
   def getObs(self):
      return self._obs
   def getMean(self):
      return self._fcst
