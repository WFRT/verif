from scipy import io
import numpy as np
import verif.util
import re
import sys
import os
import verif.input
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
class Data(object):
   def __init__(self, filenames, dates=None, offsets=None, locations=None,
         latRange=None, lonRange=None, elevRange=None, clim=None, climType="subtract",
         training=None, legend=None, removeMissingAcrossAll=True):
      if(not isinstance(filenames, list)):
         filenames = [filenames]
      self._axis = "date"
      self._index = 0
      self._removeMissingAcrossAll = removeMissingAcrossAll

      if(legend is not None and len(filenames) is not len(legend)):
         verif.util.error("Need one legend entry for each filename")
      self._legend = legend

      # Organize files
      self._files = list()
      self._cache = list()
      self._clim = None
      for filename in filenames:
         if(not os.path.exists(filename)):
            verif.util.error("File '" + filename + "' does not exist")
         if(verif.input.NetcdfCf.is_valid(filename)):
            file = verif.input.NetcdfCf(filename)
         elif(verif.input.Comps.is_valid(filename)):
            file = verif.input.Comps(filename)
         elif(verif.input.Text.is_valid(filename)):
            file = verif.input.Text(filename)
         else:
            verif.util.error("File '" + filename + "' is not a valid input file")
         self._files.append(file)
         self._cache.append(dict())
      if(clim is not None):
         if(not os.path.exists(clim)):
            verif.util.error("File '" + clim + "' does not exist")
         if(verif.input.NetcdfCf.is_valid(clim)):
            self._clim = verif.input.NetcdfCf(clim)
         elif(verif.input.Comps.is_valid(clim)):
            self._clim = verif.input.Comps(clim)
         elif(verif.input.Text.is_valid(clim)):
            self._clim = verif.input.Text(clim)
         else:
            verif.util.error("File '" + clim + "' is not a valid climatology file")
         self._cache.append(dict())
         if(not (climType == "subtract" or climType == "divide")):
            verif.util.error("Data: climType must be 'subtract' or 'divide")
         self._climType = climType

      # Climatology file
         self._files = self._files + [self._clim]

      # Latitude-Longitude range
      if(latRange is not None or lonRange is not None):
         lat = self._files[0].get_lats()
         lon = self._files[0].get_lons()
         locId = self._files[0].get_station_ids()
         latlonLocations = list()
         minLon = -180
         maxLon = 180
         minLat = -90
         maxLat = 90
         if latRange is not None:
            minLat = latRange[0]
            maxLat = latRange[1]
         if lonRange is not None:
            minLon = lonRange[0]
            maxLon = lonRange[1]
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
            verif.util.error("No available locations within lat/lon range")
      elif locations is not None:
         useLocations = locations
      else:
         useLocations = self._files[0].get_station_ids()

      # Elevation range
      if(elevRange is not None):
         stations = self._files[0].get_stations()
         minElev = elevRange[0]
         maxElev = elevRange[1]
         elevLocations = list()
         for i in range(0, len(stations)):
            currElev = float(stations[i].elev())
            id = stations[i].id()
            if(currElev >= minElev and currElev <= maxElev):
               elevLocations.append(id)
         useLocations = verif.util.intersect(useLocations, elevLocations)
         if(len(useLocations) == 0):
            verif.util.error("No available locations within elevation range")

      # Find common indicies
      self._datesI = self._get_util_indices(self._files, "Date", dates)
      self._offsetsI = self._get_util_indices(self._files, "Offset", offsets)
      self._locationsI = self._get_util_indices(self._files, "Location",
            useLocations)
      if(len(self._datesI[0]) == 0):
         verif.util.error("No valid dates selected")
      if(len(self._offsetsI[0]) == 0):
         verif.util.error("No valid offsets selected")
      if(len(self._locationsI[0]) == 0):
         verif.util.error("No valid locations selected")

      # Training
      if(training is not None):
         for f in range(0, len(self._datesI)):
            if(len(self._datesI[f]) <= training):
               verif.util.error("Training period too long for " +
                     self.get_filenames()[f] + ". Max training period is " +
                     str(len(self._datesI[f]) - 1) + ".")
            self._datesI[f] = self._datesI[f][training:]

      self._findex = 0

   # Returns flattened arrays along the set axis/index
   def get_scores(self, metrics):
      if(not isinstance(metrics, list)):
         metrics = [metrics]
      data = dict()
      valid = None
      axis = self._get_axis_index(self._axis)

      # Compute climatology, if needed
      obsFcstAvailable = ("obs" in metrics or "fcst" in metrics)
      doClim = self._clim is not None and obsFcstAvailable
      if(doClim):
         temp = self._get_score("fcst", len(self._files) - 1)
         if(self._axis == "date"):
            clim = temp[self._index, :, :].flatten()
         elif(self._axis == "month"):
            dates = self.get_axis_values("date")
            months = self.get_axis_values("month")
            if(self._index == months.shape[0]-1):
               I = np.where(dates >= months[self._index])
            else:
               I = np.where((dates >= months[self._index]) &
                            (dates < months[self._index+1]))
            clim = temp[I, :, :].flatten()
         elif(self._axis == "year"):
            dates = self.get_axis_values("date")
            years = self.get_axis_values("year")
            if(self._index == years.shape[0]-1):
               I = np.where(dates >= years[self._index])
            else:
               I = np.where((dates >= years[self._index]) &
                            (dates < years[self._index+1]))
            clim = temp[I, :, :].flatten()
         elif(self._axis == "offset"):
            clim = temp[:, self._index, :].flatten()
         elif(self.is_location_axis(self._axis)):
            clim = temp[:, :, self._index].flatten()
         elif(self._axis == "none" or self._axis == "threshold"):
            clim = temp.flatten()
         elif(self._axis == "all"):
            clim = temp
      else:
         clim = 0

      for i in range(0, len(metrics)):
         metric = metrics[i]
         temp = self._get_score(metric)
         # print self._axis

         if(self._axis == "date"):
            data[metric] = temp[self._index, :, :].flatten()
         elif(self._axis == "month"):
            dates = self.get_axis_values("date")
            months = self.get_axis_values("month")
            if(self._index == months.shape[0]-1):
               I = np.where(dates >= months[self._index])
            else:
               I = np.where((dates >= months[self._index]) &
                            (dates < months[self._index+1]))
            data[metric] = temp[I, :, :].flatten()
         elif(self._axis == "year"):
            dates = self.get_axis_values("date")
            years = self.get_axis_values("year")
            if(self._index == years.shape[0]-1):
               I = np.where(dates >= years[self._index])
            else:
               I = np.where((dates >= years[self._index]) &
                            (dates < years[self._index+1]))
            data[metric] = temp[I, :, :].flatten()
         elif(self._axis == "offset"):
            data[metric] = temp[:, self._index, :].flatten()
         elif(self.is_location_axis(self._axis)):
            data[metric] = temp[:, :, self._index].flatten()
         elif(self._axis == "none" or self._axis == "threshold"):
            data[metric] = temp.flatten()
         elif(self._axis == "all"):
            data[metric] = temp
         else:
            verif.util.error("Data.py: unrecognized value of self._axis: " +
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

   def get_stations(self):
      stations = self._files[0].get_stations()
      I = self._locationsI[0]
      useStations = list()
      for i in I:
         useStations.append(stations[i])
      return useStations

   # Find indicies of elements that are present in all files
   # Merge in values in 'aux' as well
   @staticmethod
   def _get_util_indices(files, name, aux=None):
      # Find common values among all files
      values = aux
      for file in files:
         if(name == "Date"):
            temp = file.get_dates()
         elif(name == "Offset"):
            temp = file.get_offsets()
         elif(name == "Location"):
            stations = file.get_stations()
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
            temp = file.get_dates()
         elif(name == "Offset"):
            temp = file.get_offsets()
         elif(name == "Location"):
            stations = file.get_stations()
            temp = np.zeros(len(stations))
            for i in range(0, len(stations)):
               temp[i] = stations[i].id()
         I = np.where(np.in1d(temp, values))[0]
         II = np.zeros(len(I), 'int')
         for i in range(0, len(I)):
            II[i] = np.where(values[i] == temp)[0]

         indices.append(II)
      return indices

   def _get_files(self):
      if(self._clim is None):
         return self._files
      else:
         return self._files[0:-1]

   def get_metrics(self):
      metrics = None
      for file in self._files:
         currMetrics = file.get_metrics()
         if(metrics is None):
            metrics = currMetrics
         else:
            metrics = set(metrics) & set(currMetrics)

      return metrics

   def get_thresholds(self):
      thresholds = None
      for file in self._files:
         currThresholds = file.get_thresholds()
         if(thresholds is None):
            thresholds = currThresholds
         else:
            thresholds = set(thresholds) & set(currThresholds)

      thresholds = sorted(thresholds)
      return thresholds

   def get_quantiles(self):
      quantiles = None
      for file in self._files:
         currQuantiles = file.get_quantiles()
         if(quantiles is None):
            quantiles = currQuantiles
         else:
            quantiles = set(quantiles) & set(currQuantiles)

      quantiles = sorted(quantiles)
      return quantiles

   # Get the names of all quantile scores
   def get_quantile_names(self):
      quantiles = self.get_quantiles()
      return [self.get_q_var(quantile / 100) for quantile in quantiles]

   def _get_indices(self, axis, findex=None):
      if(axis == "date"):
         I = self._get_date_indices(findex)
      elif(axis == "offset"):
         I = self._get_offset_indices(findex)
      elif(axis == "location"):
         I = self._get_location_indices(findex)
      else:
         verif.util.error("Could not get indices for axis: " + str(axis))
      return I

   def _get_date_indices(self, findex=None):
      if(findex is None):
         findex = self._findex
      return self._datesI[findex]

   def _get_offset_indices(self, findex=None):
      if(findex is None):
         findex = self._findex
      return self._offsetsI[findex]

   def _get_location_indices(self, findex=None):
      if(findex is None):
         findex = self._findex
      return self._locationsI[findex]

   def _get_score(self, metric, findex=None):
      if(findex is None):
         findex = self._findex

      if(metric in self._cache[findex]):
         return self._cache[findex][metric]

      # Load all files
      for f in range(0, self.get_num_files_with_clim()):
         if(metric not in self._cache[f]):
            file = self._files[f]
            if(metric not in file.get_variables()):
               verif.util.error("Variable '" + metric + "' does not exist in " +
                     self.get_filenames()[f])
            temp = file.get_scores(metric)
            dims = file.get_dims(metric)
            temp = verif.util.clean(temp)
            for i in range(0, len(dims)):
               I = self._get_indices(dims[i].lower(), f)
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
         for f in range(1, self.get_num_files_with_clim()):
            isMissing = isMissing | (np.isnan(self._cache[f][metric]))
         for f in range(0, self.get_num_files_with_clim()):
            self._cache[f][metric][isMissing] = np.nan

      return self._cache[findex][metric]

   # Checks that all files have the variable
   def has_metric(self, metric):
      for f in range(0, self.get_num_files_with_clim()):
         if(metric not in self._files[f].get_variables()):
            return False
      return True

   # Set the axis along which data is aggregated. One of offset, date, month,
   # year, location, lat, lon, elev.
   def set_axis(self, axis):
      self._index = 0  # Reset index
      self._axis = axis

   def set_index(self, index):
      self._index = index

   def set_file_index(self, index):
      self._findex = index

   def get_num_files(self):
      return len(self._files) - (self._clim is not None)

   def get_num_files_with_clim(self):
      return len(self._files)

   def get_units(self):
      # TODO: Only check first file?
      return self._files[0].get_units()

   def is_location_axis(self, axis):
      if(axis is None):
         return False
      return axis in ["location", "locationId", "lat", "lon", "elev"]

   def get_axis_size(self, axis=None):
      if(axis is None):
         axis = self._axis
      return len(self.get_axis_values(axis))

   # What values represent this axis?
   def get_axis_values(self, axis=None):
      if(axis is None):
         axis = self._axis
      if(axis == "date"):
         return verif.util.convert_dates(self._get_score("Date").astype(int))
      elif(axis == "month"):
         dates = self._get_score("Date").astype(int)
         months = np.unique((dates / 100) * 100 + 1)
         return verif.util.convert_dates(months)
      elif(axis == "year"):
         dates = self._get_score("Date").astype(int)
         years = np.unique((dates / 10000) * 10000 + 101)
         return verif.util.convert_dates(years)
      elif(axis == "offset"):
         return self._get_score("Offset").astype(int)
      elif(axis == "none"):
         return [0]
      elif(self.is_location_axis(axis)):
         if(axis == "location"):
            data = range(0, len(self._get_score("Location")))
         elif(axis == "locationId"):
            data = self._get_score("Location").astype(int)
         elif(axis == "elev"):
            data = self._get_score("Elev")
         elif(axis == "lat"):
            data = self._get_score("Lat")
         elif(axis == "lon"):
            data = self._get_score("Lon")
         else:
            verif.util.error("Data.get_axis_values has a bad axis name: " + axis)
         return data
      else:
         return [0]

   def is_axis_continuous(self, axis=None):
      if(axis is None):
         axis = self._axis
      return axis in ["date", "offset", "threshold", "month", "year"]

   def is_axis_date(self, axis=None):
      if(axis is None):
         axis = self._axis
      return axis in ["date", "month", "year"]

   def get_axis_formatter(self, axis=None):
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

   def get_axis_locator(self, axis=None):
      if(axis is None):
         axis = self._axis
      if(axis == "offset"):
         # Define our own locators, since in general we want multiples of 24
         # (or even fractions thereof) to make the ticks repeat each day. Aim
         # for a maximum of 12 ticks.
         offsets = self.get_axis_values("offset")
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
   def get_full_filenames(self):
      names = list()
      files = self._get_files()
      for i in range(0, len(files)):
         names.append(files[i].get_filename())
      return names

   def get_filename(self, findex=None):
      if(findex is None):
         findex = self._findex
      return self.get_filenames()[findex]

   # Do not include the path
   def get_filenames(self):
      names = self.get_full_filenames()
      for i in range(0, len(names)):
         I = names[i].rfind('/')
         names[i] = names[i][I + 1:]
      return names

   def get_legend(self):
      if(self._legend is None):
         legend = self.get_filenames()
      else:
         legend = self._legend
      return legend

   def get_short_names(self):
      names = self.get_filenames()
      for i in range(0, len(names)):
         I = names[i].rfind('.')
         names[i] = names[i][:I]
      return names

   def get_axis(self, axis=None):
      if(axis is None):
         axis = self._axis
      return axis

   def get_variable(self):
      return self._files[0].get_variable()

   def get_variable_and_units(self):
      var = self.get_variable()
      return var.name() + " (" + var.units() + ")"

   def get_axis_label(self, axis=None):
      if(axis is None):
         axis = self._axis
      if(axis == "date"):
         return "Date"
      elif(axis == "offset"):
         return "Lead time (h)"
      elif(axis == "month"):
         return "Month"
      elif(axis == "year"):
         return "Year"
      elif(axis == "elev"):
         return "Elevation (m)"
      elif(axis == "lat"):
         return "Latitude ($^o$)"
      elif(axis == "lon"):
         return "Longitude ($^o$)"
      elif(axis == "threshold"):
         return self.get_variable_and_units()

   def get_lats(self):
      return self._get_score("Lat")

   def get_lons(self):
      return self._get_score("Lon")

   def get_elevs(self):
      return self._get_score("Elev")

   def get_location_id(self):
      return self._get_score("Location")

   def get_axis_descriptions(self, axis=None, csv=False):
      if(axis is None):
         axis = self._axis
      if self.is_location_axis(axis):
         descs = list()
         ids = self._get_score("Location")
         lats = self._get_score("Lat")
         lons = self._get_score("Lon")
         elevs = self._get_score("Elev")
         if csv:
            fmt = "%d,%f,%f,%f"
         else:
            fmt = "%6d %5.2f %5.2f %5.0f"
         for i in range(0, len(ids)):
            string = fmt % (ids[i], lats[i], lons[i], elevs[i])
            descs.append(string)
         return descs
      if(self.is_axis_date(axis)):
         values = self.get_axis_values(axis)
         values = num2date(values)
         dates = list()
         for i in range(0, len(values)):
            dates = dates + [values[i].strftime("%Y/%m/%d")]
         return dates
      else:
         return self.get_axis_values(axis)

   def get_axis_description_header(self, axis=None, csv=False):
      if(axis is None):
         axis = self._axis
      if self.is_location_axis(axis):
         if csv:
            fmt = "%s,%s,%s,%s"
         else:
            fmt = "%6s %5s %5s %5s"
         return fmt % ("id", "lat", "lon", "elev")
      else:
         return axis

   def _get_axis_index(self, axis):
      if(axis == "date"):
         return 0
      elif(axis == "offset"):
         return 1
      elif(axis == "location" or axis == "locationId" or axis == "elev" or
            axis == "lat" or axis == "lon"):
         return 2
      else:
         return None

   @staticmethod
   def get_p_var(threshold):
      return "p%g" % (threshold)

   def get_q_var(self, quantile):
      quantile = quantile * 100
      minus = ""
      if(abs(quantile - int(quantile)) > 0.01):
         var = "q" + minus + str(abs(quantile)).replace(".", "")
      else:
         var = "q" + minus + str(int(abs(quantile)))

      if(not self.has_metric(var) and quantile == 50):
         verif.util.warning("Could not find q50, using fcst instead")
         return "fcst"
      return var
