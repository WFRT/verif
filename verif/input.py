import os
import csv
import numpy as np
try:
   from netCDF4 import Dataset as netcdf
except:
   from scipy.io.netcdf import netcdf_file as netcdf
import verif.station
import verif.util
import verif.variable


class Input(object):
   """
   Abstract base class representing verification data
   """
   _description = ""    # Overwrite this

   def get_dates(self):
      pass

   def get_stations(self):
      # Returns a list of Station objects available
      pass

   def get_offsets(self):
      pass

   def get_thresholds(self):
      pass

   def get_quantiles(self):
      pass

   # Returns a 3D numpy array of observations with dims [date, offset, loc]
   def get_obs(self):
      pass

   # Returns a 3D numpy array of forecasts with dims [date, offset, loc]
   def get_deterministic(self):
      pass

   def get_metrics(self):
      pass

   def get_variables(self):
      pass

   def get_units(self):
      var = self.get_variable()
      return var.units()

   def get_variable(self):
      pass

   def get_lats(self):
      stations = self.get_stations()
      lats = np.zeros(len(stations))
      for i in range(0, len(stations)):
         lats[i] = stations[i].lat()
      return lats

   def get_lons(self):
      stations = self.get_stations()
      lons = np.zeros(len(stations))
      for i in range(0, len(stations)):
         lons[i] = stations[i].lon()
      return lons

   def get_station_ids(self):
      stations = self.get_stations()
      ids = np.zeros(len(stations))
      for i in range(0, len(stations)):
         ids[i] = stations[i].id()
      return ids

   @classmethod
   def description(cls):
      return cls._description


class Comps(Input):
   """
   Original fileformat used by OutputVerif in COMPS
   """
   _dimensionNames = ["Date", "Offset", "Location", "Lat", "Lon", "Elev"]
   _description = verif.util.format_argument("netcdf", "Undocumented legacy " +
         "NetCDF format, to be phased out. A new NetCDF based format will " +
         "be defined.")

   def __init__(self, filename):
      self._filename = os.path.expanduser(filename)
      self._file = netcdf(self._filename, 'r')

   def get_stations(self):
      lat = verif.util.clean(self._file.variables["Lat"])
      lon = verif.util.clean(self._file.variables["Lon"])
      id = verif.util.clean(self._file.variables["Location"])
      elev = verif.util.clean(self._file.variables["Elev"])
      stations = list()
      for i in range(0, lat.shape[0]):
         station = verif.station.Station(id[i], lat[i], lon[i], elev[i])
         stations.append(station)
      return stations

   def get_scores(self, metric):
      metric = self._to_p_var_comps(metric)
      temp = verif.util.clean(self._file.variables[metric])
      return temp

   def _to_p_var_verif(self, metric):
      if(metric[0] == "p" and metric != "pit"):
         metric = metric.replace("m", "-")
         if(metric != "p0"):
            metric = metric.replace("p0", "p0.")
         metric = metric.replace("p-0", "p-0.")
      return metric

   def _to_p_var_comps(self, metric):
      if(metric[0] == "p" and metric != "pit"):
         metric = metric.replace("-", "m")
         metric = metric.replace(".", "")
      return metric

   def get_dims(self, metric):
      metric = self._to_p_var_comps(metric)
      return self._file.variables[metric].dimensions

   def get_dates(self):
      return verif.util.clean(self._file.variables["Date"])

   def get_offsets(self):
      return verif.util.clean(self._file.variables["Offset"])

   def get_thresholds(self):
      thresholds = list()
      for (metric, v) in self._file.variables.iteritems():
         if(metric not in self._dimensionNames):
            if(metric[0] == "p" and metric != "pit"):
               metric = self._to_p_var_verif(metric)
               thresholds.append(float(metric[1:]))
      return thresholds

   def get_quantiles(self):
      quantiles = list()
      for (metric, v) in self._file.variables.iteritems():
         if(metric not in self._dimensionNames):
            if(metric[0] == "q"):
               quantiles.append(float(metric[1:]))
      return quantiles

   def get_metrics(self):
      metrics = list()
      for (metric, v) in self._file.variables.iteritems():
         if(metric not in self._dimensionNames):
            metrics.append(metric)
      return metrics

   def get_variables(self):
      metrics = list()
      for (metric, v) in self._file.variables.iteritems():
         metrics.append(metric)
      for i in range(0, len(metrics)):
         metrics[i] = self._to_p_var_verif(metrics[i])
      return metrics

   def get_variable(self):
      name = self._file.Variable
      units = "No units"
      if(hasattr(self._file, "Units")):
         if(self._file.Units == ""):
            units = "No units"
         elif(self._file.Units == "%"):
            units = "%"
         else:
            units = "$" + self._file.Units + "$"
      return verif.variable.Variable(name, units)

   @staticmethod
   def is_valid(filename):
      try:
         file = netcdf(filename, 'r')
      except:
         return False
      return True


class NetcdfCf(Input):
   """
   New standard format, based on NetCDF/CF
   """
   def __init__(self, filename):
      self._filename = os.path.expanduser(filename)
      self._file = netcdf(self._filename, 'r')

   @staticmethod
   def is_valid(filename):
      try:
         file = netcdf(filename, 'r')
      except:
         return False
      valid = False
      if(hasattr(file, "Conventions")):
         if(file.Conventions == "verif_1.0.0"):
            valid = True
      file.close()
      return valid

   def get_stations(self):
      lat = verif.util.clean(self._file.variables["lat"])
      lon = verif.util.clean(self._file.variables["lon"])
      id = verif.util.clean(self._file.variables["id"])
      elev = verif.util.clean(self._file.variables["elev"])
      stations = list()
      for i in range(0, lat.shape[0]):
         station = verif.station.Station(id[i], lat[i], lon[i], elev[i])
         stations.append(station)
      return stations

   def get_scores(self, metric):
      temp = verif.util.clean(self._file.variables[metric])
      return temp

   def get_obs(self):
      return verif.util.clean(self._file.variables["obs"])

   def get_fcst(self):
      return verif.util.clean(self._file.variables["fcst"])

   def get_ens(self):
      return verif.util.clean(self._file.variables["ens"])

   def get_cdf(self, threshold):
      # thresholds = get_thresholds()
      # I = np.where(thresholds == threshold)[0]
      # assert(len(I) == 1)
      temp = verif.util.clean(self._file.variables["cdf"])
      return temp

   def get_dims(self, metric):
      return self._file.variables[metric].dimensions

   def get_dates(self):
      return verif.util.clean(self._file.variables["date"])

   def get_offsets(self):
      return verif.util.clean(self._file.variables["offset"])

   def get_thresholds(self):
      return verif.util.clean(self._file.variables["thresholds"])

   def get_quantiles(self):
      return verif.util.clean(self._file.variables["quantiles"])

   def get_metrics(self):
      metrics = list()
      for (metric, v) in self._file.variables.iteritems():
         if(metric not in ["date", "offset", "id", "lat", "lon", "elev"]):
            metrics.append(metric)
      return metrics

   def get_variables(self):
      metrics = list()
      for (metric, v) in self._file.variables.iteritems():
         metrics.append(metric)
      return metrics

   def get_variable(self):
      name = self._file.standard_name
      units = "No units"
      if(hasattr(self._file, "Units")):
         if(self._file.Units == ""):
            units = "No units"
         elif(self._file.Units == "%"):
            units = "%"
         else:
            units = "$" + self._file.Units + "$"
      return verif.variable.Variable(name, units)


# Flat text file format
class Text(Input):
   _description = verif.util.format_argument("text", "Data organized in rows and columns with space as a delimiter. Each row represents one forecast/obs pair, and each column represents one attribute of the data. Here is an example:") + "\n"\
   + verif.util.format_argument("", "") + "\n"\
   + verif.util.format_argument("", "# variable: Temperature") + "\n"\
   + verif.util.format_argument("", "# units: $^oC$") + "\n"\
   + verif.util.format_argument("", "date     offset id      lat     lon      elev obs fcst      p10") + "\n"\
   + verif.util.format_argument("", "20150101 0      214     49.2    -122.1   92 3.4 2.1     0.91") + "\n"\
   + verif.util.format_argument("", "20150101 1      214     49.2    -122.1   92 4.7 4.2      0.85") + "\n"\
   + verif.util.format_argument("", "20150101 0      180     50.3    -120.3   150 0.2 -1.2 0.99") + "\n"\
   + verif.util.format_argument("", "") + "\n"\
   + verif.util.format_argument("", " Any lines starting with '#' can be metadata (currently variable: and units: are recognized). After that is a header line that must describe the data columns below. The following attributes are recognized: date (in YYYYMMDD), offset (in hours), id (station identifier), lat (in degrees), lon (in degrees), obs (observations), fcst (deterministic forecast), p<number> (cumulative probability at a threshold of 10). obs and fcst are required columns: a value of 0 is used for any missing column. The columns can be in any order. If 'id' is not provided, then they are assigned sequentially starting at 0. If there is conflicting information (for example different lat/lon/elev for the same id), then the information from the first row containing id will be used.")

   def __init__(self, filename):
      self._filename = os.path.expanduser(filename)
      file = open(self._filename, 'rU')
      self._units = "Unknown units"
      self._variable = "Unknown"
      self._pit = None

      self._dates = set()
      self._offsets = set()
      self._stations = set()
      self._quantiles = set()
      self._thresholds = set()
      fields = dict()
      obs = dict()
      fcst = dict()
      cdf = dict()
      pit = dict()
      x = dict()
      indices = dict()
      header = None

      # Default values if columns not available
      offset = 0
      date = 0
      lat = 0
      lon = 0
      elev = 0
      # Store station data, to ensure we don't have conflicting lat/lon/elev info for the same ids
      stationInfo = dict()
      shownConflictingWarning = False

      import time
      start = time.time()
      # Read the data into dictionary with (date,offset,lat,lon,elev) as key and obs/fcst as values
      for rowstr in file:
         if(rowstr[0] == "#"):
            curr = rowstr[1:]
            curr = curr.split()
            if(curr[0] == "variable:"):
               self._variable = ' '.join(curr[1:])
            elif(curr[0] == "units:"):
               self._units = curr[1]
            else:
               verif.util.warning("Ignoring line '" + rowstr.strip() + "' in file '" + self_filename + "'")
         else:
            row = rowstr.split()
            if(header is None):
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
                  else:
                     indices[att] = i

               # Ensure we have required columns
               requiredColumns = ["obs", "fcst"]
               for col in requiredColumns:
                  if(col not in indices):
                     msg = "Could not parse %s: Missing column '%s'" % (self._filename, col)
                     verif.util.error(msg)
            else:
               if(len(row) is not len(header)):
                  verif.util.error("Incorrect number of columns (expecting %d) in row '%s'"
                        % (len(header), rowstr.strip()))
               if("date" in indices):
                  date = self._clean(row[indices["date"]])
               self._dates.add(date)
               if("offset" in indices):
                  offset = self._clean(row[indices["offset"]])
               self._offsets.add(offset)
               if("id" in indices):
                  id = self._clean(row[indices["id"]])
               else:
                  id = np.nan

               # Lookup previous stationInfo
               currLat = np.nan
               currLon = np.nan
               currElev = np.nan
               if("lat" in indices):
                  currLat = self._clean(row[indices["lat"]])
               if("lon" in indices):
                  currLon = self._clean(row[indices["lon"]])
               if("elev" in indices):
                  currElev = self._clean(row[indices["elev"]])
               if not np.isnan(id) and id in stationInfo:
                  lat = stationInfo[id].lat()
                  lon = stationInfo[id].lon()
                  elev = stationInfo[id].elev()
                  if not shownConflictingWarning:
                     if (not np.isnan(currLat) and abs(currLat - lat) > 0.0001) or (not np.isnan(currLon) and abs(currLon - lon) > 0.0001) or (not np.isnan(currElev) and abs(currElev - elev) > 0.001):
                        print currLat - lat, currLon - lon, currElev - elev
                        verif.util.warning("Conflicting lat/lon/elev information: (%f,%f,%f) does not match (%f,%f,%f)" % (currLat, currLon, currElev, lat, lon, elev))
                        shownConflictingWarning = True
               else:
                  if np.isnan(currLat):
                     currLat = 0
                  if np.isnan(currLon):
                     currLon = 0
                  if np.isnan(currElev):
                     currElev = 0
                  station = verif.station.Station(id, currLat, currLon, currElev)
                  self._stations.add(station)
                  stationInfo[id] = station

               lat = stationInfo[id].lat()
               lon = stationInfo[id].lon()
               elev = stationInfo[id].elev()
               key = (date, offset, lat, lon, elev)
               obs[key] = self._clean(row[indices["obs"]])
               fcst[key] = self._clean(row[indices["fcst"]])
               quantileFields = self._get_quantile_fields(header)
               thresholdFields = self._get_threshold_fields(header)
               if "pit" in indices:
                  pit[key] = self._clean(row[indices["pit"]])
               for field in quantileFields:
                  quantile = float(field[1:])
                  self._quantiles.add(quantile)
                  key = (date, offset, lat, lon, elev, quantile)
                  x[key] = self._clean(row[indices[field]])
               for field in thresholdFields:
                  threshold = float(field[1:])
                  self._thresholds.add(threshold)
                  key = (date, offset, lat, lon, elev, threshold)
                  cdf[key] = self._clean(row[indices[field]])

      end = time.time()
      file.close()
      self._dates = list(self._dates)
      self._offsets = list(self._offsets)
      self._stations = list(self._stations)
      self._quantiles = list(self._quantiles)
      self._thresholds = np.array(list(self._thresholds))
      Ndates = len(self._dates)
      Noffsets = len(self._offsets)
      Nlocations = len(self._stations)
      Nquantiles = len(self._quantiles)
      Nthresholds = len(self._thresholds)

      # Put the dictionary data into a regular 3D array
      self._obs = np.zeros([Ndates, Noffsets, Nlocations], 'float') * np.nan
      self._fcst = np.zeros([Ndates, Noffsets, Nlocations], 'float') * np.nan
      if(len(pit) != 0):
         self._pit = np.zeros([Ndates, Noffsets, Nlocations], 'float') * np.nan
      self._cdf = np.zeros([Ndates, Noffsets, Nlocations, Nthresholds], 'float') * np.nan
      self._x = np.zeros([Ndates, Noffsets, Nlocations, Nquantiles], 'float') * np.nan
      for d in range(0, len(self._dates)):
         date = self._dates[d]
         end = time.time()
         for o in range(0, len(self._offsets)):
            offset = self._offsets[o]
            for s in range(0, len(self._stations)):
               station = self._stations[s]
               lat = station.lat()
               lon = station.lon()
               elev = station.elev()
               key = (date, offset, lat, lon, elev)
               if(key in obs):
                  self._obs[d][o][s] = obs[key]
               if(key in fcst):
                  self._fcst[d][o][s] = fcst[key]
               if(key in pit):
                  self._pit[d][o][s] = pit[key]
               for q in range(0, len(self._quantiles)):
                  quantile = self._quantiles[q]
                  key = (date, offset, lat, lon, elev, quantile)
                  if(key in x):
                     self._x[d, o, s, q] = x[key]
               for t in range(0, len(self._thresholds)):
                  threshold = self._thresholds[t]
                  key = (date, offset, lat, lon, elev, threshold)
                  if(key in cdf):
                     self._cdf[d, o, s, t] = cdf[key]
      end = time.time()
      maxStationId = np.nan
      for station in self._stations:
         if(np.isnan(maxStationId)):
            maxStationId = station.id()
         elif(station.id() > maxStationId):
            maxStationId = station.id()

      counter = 0
      if(not np.isnan(maxStationId)):
         counter = maxStationId + 1

      for station in self._stations:
         if(np.isnan(station.id())):
            station.id(counter)
            counter = counter + 1
      self._dates = np.array(self._dates)
      self._offsets = np.array(self._offsets)

   # Parse string into float, changing -999 into np.nan
   def _clean(self, value):
      fvalue = float(value)
      if(fvalue == -999):
         fvalue = np.nan
      return fvalue

   def _get_quantile_fields(self, fields):
      quantiles = list()
      for att in fields:
         if(att[0] == "q"):
            quantiles.append(att)
      return quantiles

   def _get_threshold_fields(self, fields):
      thresholds = list()
      for att in fields:
         if(att[0] == "p" and att != "pit"):
            thresholds.append(att)
      return thresholds

   def get_thresholds(self):
      return self._thresholds

   def get_quantiles(self):
      return self._quantiles

   def get_name(self):
      return "Unknown"

   def get_stations(self):
      return self._stations

   def get_scores(self, metric):
      if(metric == "obs"):
         return self._obs
      elif(metric == "fcst"):
         return self._fcst
      elif(metric == "pit"):
         if(self._pit is None):
            verif.util.error("File does not contain 'pit'")
         return self._pit
      elif(metric[0] == "p"):
         threshold = float(metric[1:])
         I = np.where(abs(self._thresholds - threshold) < 0.0001)[0]
         if(len(I) == 0):
            verif.util.error("Cannot find " + metric)
         elif(len(I) > 1):
            verif.util.error("Could not find unique threshold: " + str(threshold))
         return self._cdf[:, :, :, I[0]]
      elif(metric[0] == "q"):
         quantile = float(metric[1:])
         I = np.where(abs(self._quantiles - quantile) < 0.0001)[0]
         if(len(I) == 0):
            verif.util.error("Cannot find " + metric)
         elif(len(I) > 1):
            verif.util.error("Could not find unique quantile: " + str(quantile))
         return self._x[:, :, :, I[0]]
      elif(metric == "Offset"):
         return self._offsets
      elif(metric == "Date"):
         return self._dates
      elif(metric == "Location"):
         stations = np.zeros(len(self._stations), 'float')
         for i in range(0, len(self._stations)):
            stations[i] = self._stations[i].id()
         return stations
      elif(metric in ["Lat", "Lon", "Elev"]):
         values = np.zeros(len(self._stations), 'float')
         for i in range(0, len(self._stations)):
            station = self._stations[i]
            if(metric == "Lat"):
               values[i] = station.lat()
            elif(metric == "Lon"):
               values[i] = station.lon()
            elif(metric == "Elev"):
               values[i] = station.elev()
         return values
      else:
         verif.util.error("Cannot find " + metric)

   def get_dims(self, metric):
      if(metric in ["Date", "Offset", "Location"]):
         return [metric]
      elif(metric in ["Lat", "Lon", "Elev"]):
         return ["Location"]
      else:
         return ["Date", "Offset", "Location"]

   def get_dates(self):
      return self._dates

   def get_offsets(self):
      return self._offsets

   def get_metrics(self):
      metrics = ["obs", "fcst"]
      for quantile in self._quantiles:
         metrics.append("q%g" % quantile)
      for threshold in self._thresholds:
         metrics.append("p%g" % threshold)
      if(self._pit is not None):
         metrics.append("pit")
      return metrics

   def get_quantiles(self):
      return self._quantiles

   def get_variables(self):
      metrics = self.get_metrics() + ["Date", "Offset", "Location", "Lat", "Lon", "Elev"]
      return metrics

   def get_variable(self):
      return verif.variable.Variable(self._variable, self._units)

   @staticmethod
   def is_valid(filename):
      return True


class Fake(Input):
   def __init__(self, obs, fcst):
      self._obs = obs
      self._fcst = fcst

   def get_obs(self):
      return self._obs

   def get_mean(self):
      return self._fcst
