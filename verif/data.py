from scipy import io
import numpy as np
import verif.util
import re
import sys
import os
import verif.input
from matplotlib.dates import *
import matplotlib.ticker
import verif.field

class Slice(object):
   def __init__(self, input_index, axis, axis_index):
      self.input_index = input_index
      self.axis = axis
      self.axis_index = axis_index

class Data(object):
   """ Organizes data from several inputs

   Access verification data from a set of Input files. Only returns data that
   is available for all files, for fair comparisons i.e if some
   dates/offsets/locations are missing.

   Arguments:
   inputs:     A list of verif.input
   dates:      Only allow these dates
   offsets:    Only allow these offsets
   locations:  A list of verif.location to allow
   clim:       Use this NetCDF file to compute anomaly. Should therefore be a
               climatological forecast. Subtract/divide the forecasts from this
               file from all forecasts and observations from the other files.
   clim_type:   'subtract', or 'divide' the climatology
   
   Instance attribute:
   dates:      A numpy array of available dates
   offsets:    A numpy array of available leadtimes
   locations:  A list of available locations
   thresholds: A numpy array of available thresholds
   quantiles:  A numpy array of available quantiles
   num_inputs: The number of inputs in the dataset
   variable:   The variable
   months:     
   years:      
   """
   def __init__(self, inputs, dates=None, offsets=None, locations=None,
         lat_range=None, lon_range=None, elev_range=None, clim=None, clim_type="subtract",
         legend=None, remove_missing_across_all=True):

      if(not isinstance(inputs, list)):
         inputs = [inputs]
      self._remove_missing_across_all = remove_missing_across_all

      if(legend is not None and len(inputs) is not len(legend)):
         verif.util.error("Need one legend entry for each filename")
      self._legend = legend

      # Organize inputs
      self._inputs = list()
      self._cache = list()
      self._clim = None
      for input in inputs:
         self._inputs.append(input)
         self._cache.append(dict())
      if(clim is not None):
         self._clim = verif.input.get_input(clim)
         self._cache.append(dict())
         if(not (clim_type == "subtract" or clim_type == "divide")):
            verif.util.error("Data: clim_type must be 'subtract' or 'divide")
         self._clim_type = clim_type

         # Climatology file
         self._inputs = self._inputs + [self._clim]

      # Latitude-Longitude range
      if(lat_range is not None or lon_range is not None):
         lat = [loc.lat for loc in self._inputs[0].locations]
         lon = [loc.lon for loc in self._inputs[0].locations]
         loc_id = [loc.id for loc in self._inputs[0].locations]
         latlon_locations = list()
         min_lon = -180
         max_lon = 180
         min_lat = -90
         max_lat = 90
         if lat_range is not None:
            min_lat = lat_range[0]
            max_lat = lat_range[1]
         if lon_range is not None:
            min_lon = lon_range[0]
            max_lon = lon_range[1]
         for i in range(0, len(lat)):
            currLat = float(lat[i])
            currLon = float(lon[i])
            if(currLat >= min_lat and currLat <= max_lat and
                  currLon >= min_lon and currLon <= max_lon):
               latlon_locations.append(loc_id[i])
         use_locationss = list()
         if(locations is not None):
            for i in range(0, len(locations)):
               currLocation = locations[i]
               if(currLocation in latlon_locations):
                  use_locationss.append(currLocation)
         else:
            use_locationss = latlon_locations
         if(len(use_locationss) == 0):
            verif.util.error("No available locations within lat/lon range")
      elif locations is not None:
         use_locationss = locations
      else:
         use_locationss = [s.id for s in self._inputs[0].locations]

      # Elevation range
      if(elev_range is not None):
         locations = self._inputs[0].locations
         min_elev = elev_range[0]
         max_elev = elev_range[1]
         elev_locations = list()
         for i in range(0, len(locations)):
            curr_elev = float(locations[i].elev())
            id = locations[i].id()
            if(curr_elev >= min_elev and curr_elev <= max_elev):
               elev_locations.append(id)
         use_locationss = verif.util.intersect(use_locationss, elev_locations)
         if(len(use_locationss) == 0):
            verif.util.error("No available locations within elevation range")

      # Find common indicies
      self._datesI = self._get_common_indices(self._inputs, "Date", dates)
      self._offsetsI = self._get_common_indices(self._inputs, "Offset", offsets)
      self._locationsI = self._get_common_indices(self._inputs, "Location", use_locationss)
      if(len(self._datesI[0]) == 0):
         verif.util.error("No valid dates selected")
      if(len(self._offsetsI[0]) == 0):
         verif.util.error("No valid offsets selected")
      if(len(self._locationsI[0]) == 0):
         verif.util.error("No valid locations selected")

      self._findex = 0

      # Load dimension information
      self.dates = self._get_dates()
      self.offsets = self._get_offsets()
      self.locations = self._get_locations()  # TODO: change to _get_locations
      self.thresholds = self._get_thresholds()  # TODO
      self.quantiles = self._get_quantiles()  # TODO
      self.variable = self._get_variable()
      self.months = self._get_months()
      self.years = self._get_years()

      self.num_inputs = self._get_num_inputs()

   def get_scores(self, fields, input_index, axis=None, axis_index=None):
      """ Retrieves scores from all files
      
      fields:      A list of fields of scores to retrieve
      input_index: Which file to pull from
      axis:       Which axis to aggregate against. If None is used, then no
                  aggregation takes place
      axis_index: Which slice along the axis to retrieve

      Returns:
      data        A dictionary of field:np.array
      """

      data = dict()
      valid = None

      if(not isinstance(fields, list)):
         fields = [fields]

      # Compute climatology, if needed
      
      # Load data and flatten along the correct dimension
      for i in range(0, len(fields)):
         field = fields[i]
         temp = self._get_score(field, input_index)

         if(axis == verif.axis.Date):
            data[field] = temp[axis_index, :, :].flatten()
         elif(axis == verif.axis.Month):
            if(axis_index == self.months.shape[0] - 1):
               I = np.where(self.dates >= self.months[axis_index])
            else:
               I = np.where((self.dates >= self.months[axis_index]) &
                            (self.dates < self.months[axis_index + 1]))
            data[field] = temp[I, :, :].flatten()
         elif(axis == verif.axis.Year):
            if(axis_index == self.years.shape[0] - 1):
               I = np.where(self.dates >= self.years[axis_index])
            else:
               I = np.where((self.dates >= self.years[axis_index]) &
                            (self.dates < self.years[axis_index + 1]))
            data[field] = temp[I, :, :].flatten()
         elif(axis == verif.axis.Offset):
            data[field] = temp[:, axis_index, :].flatten()
         elif(self.is_location_axis(axis)):
            data[field] = temp[:, :, axis_index].flatten()
         elif(axis == verif.axis.No or axis == verif.axis.Threshold):
            data[field] = temp.flatten()
         elif(axis == verif.axis.All or axis is None):
            data[field] = temp
         else:
            verif.util.error("Data.py: unrecognized axis: " + axis)

         # Subtract climatology

         # Remove missing values
         if axis is not verif.axis.All:
            currValid = (np.isnan(data[field]) == 0)\
                      & (np.isinf(data[field]) == 0)
            if(valid is None):
               valid = currValid
            else:
               valid = (valid & currValid)
      if axis is not verif.axis.All:
         I = np.where(valid)

      q = list()
      for i in range(0, len(fields)):
         if axis is not verif.axis.All:
            q.append(data[fields[i]][I])
         else:
            q.append(data[fields[i]])

      # No valid data
      if(q[0].shape[0] == 0):
         for i in range(0, len(fields)):
            q[i] = np.nan * np.zeros([1], 'float')

      return q

   def _get_score(self, field, input_index):
      """ Load the field variable from input, but only include the common data
      
      Scores loaded will have the same dimension, regardless what input_index
      is used.

      field:         The type is of verif.field
      input_index:   which input to load from
      """

      # Check if data is cached
      if(field in self._cache[input_index]):
         return self._cache[input_index][field]

      # Load all inputs
      for i in range(0, self._get_num_inputs_with_clim()):
         if(field not in self._cache[i]):
            input = self._inputs[i]
            if(field not in input.get_variables()):
               verif.util.error("Variable '" + field.name() + "' does not exist in " +
                     self.get_names()[i])
            if field is verif.field.Obs:
               temp = input.obs
            elif field is verif.field.Deterministic:
               temp = input.deterministic
            elif field is verif.field.Ensemble:
               temp = input.ensemble[:,:,:,field.member]
            elif field is verif.field.Threshold:
               I = np.where(input.thresholds == field.threshold)[0]
               assert(len(I) == 1)
               temp = input.threshold_scores[:,:,:,I]
            elif field is verif.field.Quantile:
               I = np.where(input.quantiles == field.quantile)[0]
               assert(len(I) == 1)
               temp = input.quantile_scores[:,:,:,I]
            else:
               verif.util.error("Not implemented")
            temp = verif.util.clean(temp)
            Idates = self._get_date_indices(i)
            Ioffsets = self._get_offset_indices(i)
            Ilocations = self._get_location_indices(i)
            temp = temp[Idates, :, :]
            temp = temp[:, Ioffsets, :]
            temp = temp[:, :, Ilocations]
            self._cache[i][field] = temp

      # Remove missing. If one configuration has a missing value, set all
      # configurations to missing This can happen when the dates are available,
      # but have missing values
      if self._remove_missing_across_all:
         is_missing = np.isnan(self._cache[0][field])
         for i in range(1, self._get_num_inputs_with_clim()):
            is_missing = is_missing | (np.isnan(self._cache[i][field]))
         for i in range(0, self._get_num_inputs_with_clim()):
            self._cache[i][field][is_missing] = np.nan

      return self._cache[input_index][field]

   """
   # Returns flattened arrays along the set axis/index
   def get_scores2(self, metrics):
      if(not isinstance(metrics, list)):
         metrics = [metrics]
      data = dict()
      valid = None
      axis = self._get_axis_index(self._axis)

      # Compute climatology, if needed
      obsFcstAvailable = ("obs" in metrics or "fcst" in metrics)
      doClim = self._clim is not None and obsFcstAvailable
      if(doClim):
         temp = self._get_score("fcst", len(self._inputs) - 1)
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
            if(self._clim_type == "subtract"):
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
   """

   def _get_dates(self):
      dates = self._inputs[0].dates
      I = self._datesI[0]
      return np.array([dates[i] for i in I], int)

   def _get_months(self):
      months = np.unique((self.dates / 100) * 100 + 1)
      return verif.util.convert_dates(months)

   def _get_years(self):
      years = np.unique((self.dates / 10000) * 10000 + 101)
      return verif.util.convert_dates(years)

   def _get_offsets(self):
      offsets = self._inputs[0].offsets
      I = self._offsetsI[0]
      return np.array([offsets[i] for i in I], int)

   def _get_locations(self):
      locations = self._inputs[0].locations
      I = self._locationsI[0]
      use_locations = list()
      for i in I:
         use_locations.append(locations[i])
      return use_locations

   @staticmethod
   def _get_common_indices(files, name, aux=None):
      """
      Find indicies of elements that are present in all files. Merge in values
      in 'aux' as well

      Returns a list of arrays, one array for each file
      """
      # Find common values among all files
      values = aux
      for file in files:
         if(name == "Date"):
            temp = file.dates
         elif(name == "Offset"):
            temp = file.offsets
         elif(name == "Location"):
            locations = file.locations
            temp = [loc.id for loc in locations]
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
            temp = file.dates
         elif(name == "Offset"):
            temp = file.offsets
         elif(name == "Location"):
            locations = file.locations
            temp = np.zeros(len(locations))
            for i in range(0, len(locations)):
               temp[i] = locations[i].id
         I = np.where(np.in1d(temp, values))[0]
         II = np.zeros(len(I), 'int')
         for i in range(0, len(I)):
            II[i] = np.where(values[i] == temp)[0]

         indices.append(II)
      return indices

   def _get_files(self):
      if(self._clim is None):
         return self._inputs
      else:
         return self._inputs[0:-1]

   def _get_thresholds(self):
      thresholds = None
      for file in self._inputs:
         currThresholds = file.thresholds
         if(thresholds is None):
            thresholds = currThresholds
         else:
            thresholds = set(thresholds) & set(currThresholds)

      thresholds = sorted(thresholds)
      return thresholds

   def _get_quantiles(self):
      quantiles = None
      for file in self._inputs:
         currQuantiles = file.quantiles
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

   def _get_date_indices(self, input_index):
      return self._datesI[input_index]

   def _get_offset_indices(self, input_index):
      return self._offsetsI[input_index]

   def _get_location_indices(self, input_index):
      return self._locationsI[input_index]

   # Checks that all files have the variable
   def has_metric(self, metric):
      for f in range(0, self._get_num_inputs_with_clim()):
         if(metric not in self._inputs[f].get_variables()):
            return False
      return True

   def _get_num_inputs(self):
      return len(self._inputs) - (self._clim is not None)

   def _get_num_inputs_with_clim(self):
      return len(self._inputs)

   def _get_variable(self):
      # TODO: Only check first file?
      return self._inputs[0].variable

   def is_location_axis(self, axis):
      if(axis is None):
         return False
      return axis in [verif.axis.Location, verif.axis.LocationId,
            verif.axis.Lat, verif.axis.Lon, verif.axis.Elev]

   def get_axis_size(self, axis):
      return len(self.get_axis_values(axis))

   # What values represent this axis?
   def get_axis_values(self, axis):
      if(axis == verif.axis.Date):
         # TODO: Does it make sense to convert here, but not with data.dates?
         return verif.util.convert_dates(self.dates)
      elif(axis == verif.axis.Month):
         return self.months
      elif(axis ==verif.axis.Year):
         return self.years
      elif(axis ==verif.axis.Offset):
         return self.offsets
      elif(axis == verif.axis.No):
         return [0]
      elif(self.is_location_axis(axis)):
         if(axis == verif.axis.Location):
            data = range(0, len(self.locations))
         elif(axis == verif.axis.LocationId):
            data = self.get_location_ids()
         elif(axis == verif.axis.Elev):
            data = self.get_elevs()
         elif(axis == verif.axis.Lat):
            data = self.get_lats()
         elif(axis == verif.axis.Lon):
            data = self.get_lons()
         else:
            verif.util.error("Data.get_axis_values has a bad axis name: " + axis)
         return data
      else:
         return [0]

   def is_axis_continuous(self, axis):
      return axis in [verif.axis.Date, verif.axis.Offset, verif.axis.Threshold,
            verif.axis.Month, verif.axis.Year]

   def is_axis_date(self, axis):
      return axis in [verif.axis.Date, verif.axis.Month, verif.axis.Year]

   def get_axis_formatter(self, axis):
      if(axis == verif.axis.Date):
         return DateFormatter('\n%Y-%m-%d')
      elif(axis == verif.axis.Month):
         return DateFormatter('\n%Y-%m')
      elif(axis == verif.axis.Year):
         return DateFormatter('\n%Y')
      else:
         return matplotlib.ticker.ScalarFormatter()

   def get_axis_locator(self, axis):
      if(axis == verif.axis.Offset):
         # Define our own locators, since in general we want multiples of 24
         # (or even fractions thereof) to make the ticks repeat each day. Aim
         # for a maximum of 12 ticks.
         offsets = self.get_axis_values(verif.axis.Offset)
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
   def get_full_names(self):
      names = [input.fullname for input in self._inputs]
      return names

   def get_name(self, input_index):
      return self.get_filenames()[input_index]

   # Do not include the path
   def get_names(self):
      names = [input.name for input in self._inputs]
      return names

   def get_short_names(self):
      return [input.shortname for input in inputs]

   def get_legend(self):
      if(self._legend is None):
         legend = self.get_names()
      else:
         legend = self._legend
      return legend

   def get_variable_and_units(self):
      var = self.variable
      return var.name + " (" + var.units + ")"

   def get_axis_label(self, axis):
      if(axis == verif.axis.Date):
         return "Date"
      elif(axis == verif.axis.Offset):
         return "Lead time (h)"
      elif(axis == verif.axis.Month):
         return "Month"
      elif(axis == verif.axis.Year):
         return "Year"
      elif(axis == verif.axis.Elev):
         return "Elevation (m)"
      elif(axis == verif.axis.Lat):
         return "Latitude ($^o$)"
      elif(axis == verif.axis.Lon):
         return "Longitude ($^o$)"
      elif(axis == verif.axis.Threshold):
         return self.get_variable_and_units()

   def get_lats(self):
      return np.array([loc.lat for loc in self.locations])

   def get_lons(self):
      return np.array([loc.lon for loc in self.locations])

   def get_elevs(self):
      return np.array([loc.elev for loc in self.locations])

   def get_location_ids(self):
      return np.array([loc.id for loc in self.locations], int)

   def get_axis_descriptions(self, axis, csv=False):
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

   def get_axis_description_header(self, axis, csv=False):
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
