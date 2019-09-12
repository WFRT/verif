import numpy as np
import re
import sys
import os
import datetime
import time
import calendar
import verif.input
import matplotlib.dates
import matplotlib.ticker
import verif.field
import verif.util
import verif.variable


class Data(object):
    """ Organizes data from several inputs

    Access verification data from a list of verif.input. Only returns data that
    is available for all files, for fair comparisons i.e if some
    times/leadtimes/locations are missing.

    Instance attribute:
    times          A numpy array of available initialization times
    leadtimes      A numpy array of available leadtimes
    locations      A list of available locations
    thresholds     A numpy array of available thresholds
    quantiles      A numpy array of available quantiles
    num_inputs     The number of inputs in the dataset
    variable       The variable
    timesofday     Available times of day (derived from times)
    daysofyear     Available days of year (derived from times)
    days           Available days (derived from times)
    weeks          Available weeks (derived from times)
    months         Available months (derived from times)
    years          Available years (derived from times)
    leadtimedays   Available leadtimedays (derived from leadtimes)
    """
    def __init__(self, inputs, times=None, dates=None, tods=None, leadtimes=None, locations=None, locations_x=None,
          lat_range=None, lon_range=None, elev_range=None, clim=None, clim_type="subtract",
          obs_range=None, legend=None, remove_missing_across_all=True,
          obs_field=verif.field.Obs(),
          fcst_field=verif.field.Fcst()):

        """
        Arguments:
        inputs         A list of verif.input
        times          A numpy array of times. Discard data for all other times
        dates          A numpy array of dates. Only allow times that occur on
                       these dates (but alllow any hour of the day).
        tods           A numpy array of time of day values (in hours). Only allow
                       initialization times for these hours of the day (in UTC).
        leadtimes      A numpy array of leadtimes. Discard data for all other leadtimes
        locations      A list of verif.location. Discard data for all other locations
        locations_x    A list of verif.location to not remove
        clim           Use this verif.input to compute anomaly. Should therefore
                       be a climatological forecast. Subtract/divide the
                       forecasts from this file from all forecasts and
                       observations from the other files.
        clim_type      Operation to apply with climatology. Either 'subtract', or
                       'divide'
        """

        if not isinstance(inputs, list):
            inputs = [inputs]
        self._remove_missing_across_all = remove_missing_across_all

        if legend is not None and len(inputs) is not len(legend):
            verif.util.error("Need one legend entry for each filename")
        self._legend = legend

        self._obs_field = obs_field
        self._fcst_field = fcst_field

        self._obs_range = obs_range

        # Organize inputs
        self._inputs = list()
        self._get_score_cache = list()  # Caches data from input
        self._get_scores_cache = dict()  # Caches the output from get_scores
        self._clim = None
        for input in inputs:
            self._inputs.append(input)
            self._get_score_cache.append(dict())
        if clim is not None:
            self._clim = clim
            self._get_score_cache.append(dict())
            if not (clim_type == "subtract" or clim_type == "divide"):
                verif.util.error("Data: clim_type must be 'subtract' or 'divide")
            self._clim_type = clim_type

            # Add climatology to the end
            self._inputs = self._inputs + [self._clim]

        # Latitude-Longitude range
        if lat_range is not None or lon_range is not None:
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
                if currLat >= min_lat and currLat <= max_lat and currLon >= min_lon and currLon <= max_lon:
                    latlon_locations.append(loc_id[i])
            use_locations = list()
            if locations is not None:
                use_locations = [loc for loc in locations if loc in latlon_locations]
            else:
                use_locations = latlon_locations
            if len(use_locations) == 0:
                verif.util.error("No available locations within lat/lon range")
        elif locations is not None:
            use_locations = locations
        else:
            use_locations = [s.id for s in self._inputs[0].locations]

        # Elevation range
        if elev_range is not None:
            locations = self._inputs[0].locations
            min_elev = elev_range[0]
            max_elev = elev_range[1]
            elev_locations = list()
            for i in range(0, len(locations)):
                curr_elev = float(locations[i].elev)
                id = locations[i].id
                if curr_elev >= min_elev and curr_elev <= max_elev:
                    elev_locations.append(id)
            use_locations = verif.util.intersect(use_locations, elev_locations)
            if len(use_locations) == 0:
                verif.util.error("No available locations within elevation range")

        # Remove locations
        if locations_x is not None:
            use_locations = [loc for loc in use_locations if loc not in locations_x]

        # Find common indicies
        # TODO: Merge in information from dates (but these span whole days)
        self._timesI = self._get_common_indices(self._inputs, verif.axis.Time(), times)
        self._leadtimesI = self._get_common_indices(self._inputs, verif.axis.Leadtime(), leadtimes)
        self._locationsI = self._get_common_indices(self._inputs, verif.axis.Location(), use_locations)
        if len(self._timesI[0]) == 0:
            verif.util.error("No valid times selected")
        if len(self._leadtimesI[0]) == 0:
            verif.util.error("No valid leadtimes selected")
        if len(self._locationsI[0]) == 0:
            verif.util.error("No valid locations selected")

        # Load dimension information
        self.times = self._get_times()
        self.leadtimes = self._get_leadtimes()
        self.locations = self._get_locations()
        self.thresholds = self._get_thresholds()
        self.quantiles = self._get_quantiles()
        self.variable = self._get_variable()
        self.num_inputs = self._get_num_inputs()

        if dates is not None:
            """
            Only allow specified dates. Allow all initialization times that are
            within each of the dates.
            """
            dates_times = [verif.util.date_to_unixtime(t) for t in dates]
            self.times = np.array([t for t in self.times if int(t / 86400)*86400 in dates_times])

        if tods is not None:
            self.times = np.array([t for t in self.times if int(t % 86400)/3600 in tods])
        """
        Indices must be recomputed since some times are removed by -d and -tod.
        There doesn't seem to be an easy way to avoid this.
        """
        self._timesI = self._get_common_indices(self._inputs, verif.axis.Time(), self.times)

        # Compute axis values
        self.axis_cache = dict()
        self.axis_cache_unique = dict()
        for axis in verif.axis.get_time_axes():
            self.axis_cache[axis] = axis.compute_from_times(self.times)
            self.axis_cache_unique[axis] = np.unique(self.axis_cache[axis])
        for axis in verif.axis.get_leadtime_axes():
            self.axis_cache[axis] = axis.compute_from_leadtimes(self.leadtimes)
            self.axis_cache_unique[axis] = np.unique(self.axis_cache[axis])

    def get_fields(self):
        """ Get a list of fields that all inputs have

        Returns:
           list(verif.field.Field): A list of fields
        """
        all_fields = set()
        for f in range(self._get_num_inputs_with_clim()):
            input = self._inputs[f]
            if f == 0:
                all_fields = input.get_fields()
            else:
                all_fields = set(all_fields) & set(input.get_fields())
        return list(all_fields)

    def get_scores(self, fields, input_index, axis=verif.axis.All(), axis_index=None):
        """ Retrieves scores from all files

        Climatology is handled by subtracting clim's fcst field from any
        obs or determinsitic fields.

        Arguments:
        fields         Which verif.field should be retreived? Either a list or a
                       single field can be supplied, however the return type will
                       match this choice.
        input_index    Which input to pull from? Must be between 0 and num_inputs
        axis           Which axis to aggregate against. If verif.axis.All() is
                       used, then no aggregation takes place and the 3D numpy
                       array is returned.
        axis_index     Which slice along the axis to retrieve

        Returns:
        scores         A list of numpy arrays. If fields is not a list, but a
                       single field, then a numpy array is returned.
        """

        fields_is_single = False
        if not isinstance(fields, list):
            fields = [fields]
            fields_is_single = True

        key = (tuple(fields), input_index, axis, axis_index)
        if key in self._get_scores_cache.keys():
            if fields_is_single:
                return self._get_scores_cache[key][0]
            else:
                return self._get_scores_cache[key]

        if input_index < 0 or input_index >= self.num_inputs:
            verif.util.error("input_index must be between 0 and %d" % self.num_inputs)

        scores = list()
        valid = None

        # Compute climatology, if needed
        obsFcstAvailable = (verif.field.Obs() in fields or verif.field.Fcst() in fields)
        doClim = self._clim is not None and obsFcstAvailable
        if doClim:
            temp = self._get_score(verif.field.Fcst(), len(self._inputs) - 1)
            clim = self._apply_axis(temp, axis, axis_index)
        else:
            clim = 0

        # Load scores and flatten along the correct dimension
        for i in range(0, len(fields)):
            field = fields[i]
            temp = self._get_score(field, input_index)

            # Remove observations outside the obsrange
            if self._obs_range is not None and field == verif.field.Obs():
                with np.errstate(invalid='ignore'):
                    temp[temp < self._obs_range[0]] = np.nan
                    temp[temp > self._obs_range[1]] = np.nan

            curr = self._apply_axis(temp, axis, axis_index)

            # Subtract climatology
            if doClim and (field == verif.field.Fcst() or field == verif.field.Obs()):
                if self._clim_type == "subtract":
                    curr = curr - clim
                else:
                    curr = curr / clim

            # Remove missing values
            currValid = (np.isnan(curr) == 0) & (np.isinf(curr) == 0)
            if valid is None:
                valid = currValid
            else:
                valid = (valid & currValid)
            scores.append(curr)

        if axis == verif.axis.All():
            I = np.where(valid == 0)
            for i in range(0, len(fields)):
                scores[i][I[0], I[1], I[2]] = np.nan
        else:
            I = np.where(valid)
            for i in range(0, len(fields)):
                scores[i] = scores[i][I]

        # No valid data. Therefore return a list of nans instead of an empty list
        if scores[0].shape[0] == 0:
            scores = [np.nan * np.zeros(1, float) for i in range(0, len(fields))]

        self._get_scores_cache[key] = scores

        # Turn into a single numpy array if we were not supplied with a list of
        # fields
        if fields_is_single:
            return scores[0]
        else:
            return scores

    def get_axis_size(self, axis):
        return len(self.get_axis_values(axis))

    def get_axis_values(self, axis):
        """ What are the values along an axis?
        verif.axis.Time()          Unixtimes
        verif.axis.Month()         Unixtimes of the begining of each month
        verif.axis.Year()          Unixtimes of the beginning of each year
        verif.axis.Leadtime()      Lead times in hours
        verif.axis.Leadtimeday()   Lead time day in days
        verif.axis.Timeofday()     Time of day in hours
        verif.axis.Location()      Location id
        verif.axis.Lat()           Latitudes of locations
        verif.axis.Lon()           Longitudes of locations
        verif.axis.Elev()          Elevations of locations

        Arguments:
        axis        of type verif.axis.Axis

        Returns:
        array       a 1D numpy array of values
        """
        if axis == verif.axis.Time():
            return self.times
        elif axis in verif.axis.get_time_axes():
            return np.unique(axis.compute_from_times(self.times))
        elif axis in verif.axis.get_leadtime_axes():
            return np.unique(axis.compute_from_leadtimes(self.leadtimes))
        elif axis == verif.axis.No():
            return [0]
        elif axis.is_location_like:
            if axis == verif.axis.Location():
                data = np.array([loc.id for loc in self.locations])
            elif axis == verif.axis.Elev():
                data = np.array([loc.elev for loc in self.locations])
            elif axis == verif.axis.Lat():
                data = np.array([loc.lat for loc in self.locations])
            elif axis == verif.axis.Lon():
                data = np.array([loc.lon for loc in self.locations])
            else:
                verif.util.error("Data.get_axis_values has a bad axis name: " + axis)
            return data
        else:
            return [0]

    def get_axis_locator(self, axis):
        """ Where should ticks be located for this axis? Returns an mpl Locator """
        if axis == verif.axis.Leadtime():
            # Define our own locators, since in general we want multiples of 24
            # (or even fractions thereof) to make the ticks repeat each day. Aim
            # for a maximum of 12 ticks.
            leadtimes = self.get_axis_values(verif.axis.Leadtime())
            span = max(leadtimes) - min(leadtimes)
            if span > 300:
                return matplotlib.ticker.AutoLocator()
            elif span > 200:
                return matplotlib.ticker.MultipleLocator(48)
            elif span > 144:
                return matplotlib.ticker.MultipleLocator(24)
            elif span > 72:
                return matplotlib.ticker.MultipleLocator(12)
            elif span > 36:
                return matplotlib.ticker.MultipleLocator(6)
            elif span > 12:
                return matplotlib.ticker.MultipleLocator(3)
            else:
                return matplotlib.ticker.MultipleLocator(1)
        else:
            return matplotlib.ticker.AutoLocator()

    def get_full_names(self):
        names = [input.fullname for input in self._inputs]
        if self._clim is not None:
            names = names[0:-1]
        return names

    def get_names(self):
        names = [input.name for input in self._inputs]
        if self._clim is not None:
            names = names[0:-1]
        return names

    def get_short_names(self):
        names = [input.shortname for input in self._inputs]
        if self._clim is not None:
            names = names[0:-1]
        return names

    def get_legend(self):
        if self._legend is None:
            legend = self.get_names()
        else:
            legend = self._legend
        return legend

    def get_variable_and_units(self):
        var = self.variable
        return var.name + " (" + var.units + ")"

    def get_axis_descriptions(self, axis):
        if axis.is_location_like:
            q = dict()
            descs = list()
            ids = [loc.id for loc in self.locations]
            lats = [loc.lat for loc in self.locations]
            lons = [loc.lon for loc in self.locations]
            elevs = [loc.elev for loc in self.locations]
            return {"id": ids, "lat": lats, "lon": lons, "elev": elevs}
        elif axis.is_time_like:
            unixtimes = self.get_axis_values(axis)
            # Convert to date objects
            dates = [matplotlib.dates.num2date(verif.util.unixtime_to_datenum(unixtime)) for unixtime in unixtimes]
            times = list()
            fmt = axis.fmt

            times = [date.strftime(fmt) for date in dates]
            return {axis.name(): times}
        else:
            return {axis.name(): self.get_axis_values(axis)}

    def _get_score(self, field, input_index):
        """ Load the field variable from input, but only include the common data

        Scores loaded will have the same dimension, regardless what input_index
        is used.

        field:         The type is of verif.field
        input_index:   which input to load from
        """

        num_inputs = self._get_num_inputs_with_clim()
        if field == verif.field.Obs():
            """
            Treat observation different. Since the observations should be the same
            for all inputs, reuse the observations if one or more inputs are
            missing these
            """
            field = self._obs_field
            if field in self._get_score_cache[input_index]:
                return self._get_score_cache[input_index][field]

            found_obs = False
            for i in range(num_inputs):
                input = self._inputs[i]
                all_fields = input.get_fields()
                if field in all_fields:
                    if field == verif.field.Obs():
                        temp = input.obs
                    elif field == verif.field.Fcst():
                        temp = input.fcst
                    else:
                        temp = input.other_score(field.name())
                    Itimes = self._get_time_indices(i)
                    Ileadtimes = self._get_leadtime_indices(i)
                    Ilocations = self._get_location_indices(i)
                    temp = temp[Itimes, :, :]
                    temp = temp[:, Ileadtimes, :]
                    temp = temp[:, :, Ilocations]
                    self._get_score_cache[i][field] = temp
                    found_obs = True
            if not found_obs:
                verif.util.error("No files have observations")

            for i in range(num_inputs):
                if field not in self._get_score_cache[i]:
                    for j in range(num_inputs):
                        if field in self._get_score_cache[j]:
                            verif.util.warning("No observations in %s. Loading from %s" % (self._inputs[i].fullname, self._inputs[j].fullname))
                            self._get_score_cache[i][field] = self._get_score_cache[j][field]
                            break
        else:
            # Check if data is cached
            if field in self._get_score_cache[input_index]:
                return self._get_score_cache[input_index][field]

            if field == verif.field.Fcst():
                field = self._fcst_field

            for i in range(num_inputs):
                if field not in self._get_score_cache[i]:
                    input = self._inputs[i]
                    all_fields = input.get_fields()
                    if field not in all_fields:
                        verif.util.error("%s does not contain '%s'" % (self.get_names()[i], field.name()))

                    elif field == verif.field.Obs():
                        temp = input.obs

                    elif field == verif.field.Fcst():
                        temp = input.fcst

                    elif field == verif.field.Pit():
                        temp = input.pit
                        x0 = self.variable.x0
                        x1 = self.variable.x1
                        if x0 is not None or x1 is not None:
                            temp = verif.field.Pit.randomize(input.obs, temp, x0, x1)

                    elif field.__class__ is verif.field.Ensemble:
                        temp = input.ensemble[:, :, :, field.member]

                    elif field.__class__ is verif.field.Threshold:
                        I = np.where(np.isclose(input.thresholds,  field.threshold))[0]
                        assert(len(I) == 1)
                        temp = input.threshold_scores[:, :, :, I[0]]

                    elif field.__class__ is verif.field.Quantile:
                        I = np.where(np.isclose(input.quantiles, field.quantile))[0]
                        assert(len(I) == 1)
                        temp = input.quantile_scores[:, :, :, I[0]]

                    else:
                        temp = input.other_score(field.name())
                    Itimes = self._get_time_indices(i)
                    Ileadtimes = self._get_leadtime_indices(i)
                    Ilocations = self._get_location_indices(i)
                    temp = temp[Itimes, :, :]
                    temp = temp[:, Ileadtimes, :]
                    temp = temp[:, :, Ilocations]
                    self._get_score_cache[i][field] = temp

        """
        Remove missing. If one configuration has a missing value, set all
        configurations to missing. This can happen when the times are
        available, but have missing values.
        """
        if self._remove_missing_across_all:
            is_missing = np.isnan(self._get_score_cache[0][field])
            for i in range(1, num_inputs):
                is_missing = is_missing | (np.isnan(self._get_score_cache[i][field]))
            for i in range(num_inputs):
                self._get_score_cache[i][field][is_missing] = np.nan

        return self._get_score_cache[input_index][field]

    def _calculate_window(self, array, leadtimes):
        O = array.shape[1]
        Inan = np.isnan(array)
        for o in range(0, O):
            threshold = 0.5
            q = np.nansum(np.cumsum(array[:, o:, :], axis=1) <= threshold, axis=1)
            I = q+o
            I[I >= O] = O-1
            array[:, o, :] = leadtimes[I] - leadtimes[o]
        array[Inan] = np.nan
        # array[array > 2] = 2
        return array

    def _get_times(self):
        times = self._inputs[0].times
        I = self._timesI[0]
        return np.array([times[i] for i in I], int)

    def _get_leadtimes(self):
        leadtimes = self._inputs[0].leadtimes
        I = self._leadtimesI[0]
        return np.array([leadtimes[i] for i in I], float)

    def _get_locations(self):
        locations = self._inputs[0].locations
        I = self._locationsI[0]
        use_locations = list()
        for i in I:
            use_locations.append(locations[i])
        return use_locations

    @staticmethod
    def _get_common_indices(inputs, axis, aux=None):
        """
        Find indicies of elements that are present in all inputs. Merge in values
        in 'aux' as well

        Arguments:
        inputs      A list of verif.input
        axis        An axis of type verif.axis

        Returns a list of arrays, one array for each input
        """
        # Find common values among all inputs
        available_values = aux
        for input in inputs:
            if axis == verif.axis.Time():
                curr_values = input.times
            elif axis == verif.axis.Leadtime():
                curr_values = input.leadtimes
            elif axis == verif.axis.Location():
                locations = input.locations
                curr_values = [loc.id for loc in locations]

            curr_values = np.sort(curr_values)
            len_before = len(curr_values)
            curr_values = np.unique(curr_values)

            if len(curr_values) != len_before:
                verif.util.warning("The '%s' axis in '%s' has repeated values.  Using the first value." % (axis.name(), input.fullname))

            if available_values is None:
                available_values = curr_values
            else:
                available_values = np.intersect1d(available_values, curr_values)

        # Sort values, since for example, times may not be in an ascending order
        available_values = np.sort(available_values)

        # Remove nan values
        available_values = available_values[np.isnan(available_values) == 0]

        # Determine which index each value is at
        indices = list()
        for input in inputs:
            if axis == verif.axis.Time():
                temp = input.times
            elif axis == verif.axis.Leadtime():
                temp = input.leadtimes
            elif axis == verif.axis.Location():
                locations = input.locations
                temp = np.zeros(len(locations))
                for i in range(0, len(locations)):
                    temp[i] = locations[i].id
            I = np.where(np.in1d(temp, available_values))[0]
            II = np.zeros(len(available_values), 'int')
            for i in range(0, len(available_values)):
                III = np.where(available_values[i] == temp)[0]
                assert(len(III) >= 1)
                II[i] = III[0]

            indices.append(II)
        return indices

    def _get_thresholds(self):
        thresholds = None
        for input in self._inputs:
            currThresholds = input.thresholds
            if thresholds is None:
                thresholds = currThresholds
            else:
                thresholds = set(thresholds) & set(currThresholds)

        thresholds = sorted(thresholds)
        return thresholds

    def _get_quantiles(self):
        quantiles = None
        for input in self._inputs:
            currQuantiles = input.quantiles
            if quantiles is None:
                quantiles = currQuantiles
            else:
                quantiles = set(quantiles) & set(currQuantiles)

        quantiles = sorted(quantiles)
        return quantiles

    def _get_time_indices(self, input_index):
        return self._timesI[input_index]

    def _get_leadtime_indices(self, input_index):
        return self._leadtimesI[input_index]

    def _get_location_indices(self, input_index):
        return self._locationsI[input_index]

    def _get_num_inputs(self):
        return len(self._inputs) - (self._clim is not None)

    def _get_num_inputs_with_clim(self):
        return len(self._inputs)

    def _get_variable(self):
        # TODO: Only check first file?
        variable = self._inputs[0].variable

        # Handle obs field
        units = self._obs_field.units(variable)
        name = self._obs_field.label(variable)
        formatter = self._obs_field.formatter(variable)
        x0 = variable.x0
        x1 = variable.x1
        variable = verif.variable.Variable(name, units, formatter, x0, x1)

        return variable

    def _apply_axis(self, array, axis, axis_index):
        """ Slice an array along a certain axis and return an extract array

        Arguments:
        array       3D numpy array
        axis        Of type verif.axis.Axis
        axis_index  Index along the axis to slice
        """
        output = None
        if axis == verif.axis.Time():
            output = array[axis_index, :, :].flatten()
        elif axis in verif.axis.get_time_axes():
            axis_values = self.axis_cache[axis]
            axis_values_unique = self.axis_cache_unique[axis]
            I = np.where(axis_values == axis_values_unique[axis_index])
            output = array[I, :, :].flatten()
        elif axis in verif.axis.get_leadtime_axes():
            axis_values = self.axis_cache[axis]
            axis_values_unique = self.axis_cache_unique[axis]
            I = np.where(axis_values == axis_values_unique[axis_index])
            output = array[:, I, :].flatten()
        elif axis.is_location_like:
            output = array[:, :, axis_index].flatten()
        elif axis in [verif.axis.No(), verif.axis.Threshold(), verif.axis.Obs(), verif.axis.Fcst()]:
            output = array.flatten()
        elif axis == verif.axis.All() or axis is None:
            output = array
        else:
            verif.util.error("data.py: unrecognized axis: " + axis.name())

        return output
