import sys
import argparse
import numpy as np
import verif.input
import netCDF4
import copy
import scipy.signal


def convolve(array, window, ignore_missing, axis):
    """ Convolves the Verif array across the leadtime dimension """
    assert(len(array.shape) == 3)
    if axis == 'leadtime':
        if window > array.shape[1]:
            verif.util.error("Window (%d) is longer than dimension size (%d)" % (window, array.shape[1]))
        c = np.ones([1, window, 1])
        if ignore_missing:
            # Does this work?
            array[np.isnan(array)] = 0
        new_array = np.nan*np.zeros(array.shape)
        new_array[:, (window-1):, :] = scipy.signal.convolve(array, c, "valid")
    elif axis == 'time':
        if window > array.shape[0]:
            verif.util.error("Window (%d) is longer than dimension size (%d)" % (window, array.shape[0]))
        c = np.ones([window, 1, 1])
        if ignore_missing:
            # Does this work?
            array[np.isnan(array)] = 0
        new_array = np.nan*np.zeros(array.shape)
        new_array[(window-1):, :, :] = scipy.signal.convolve(array, c, "valid")
    else:
        verif.util.error('Invalid axis %s' % axis)
    return new_array


def main():
    parser = argparse.ArgumentParser(prog="accumulate", description="Accumlates a verif file. For each leadtime, sum up values in a specified number of leadtimes leading up to this time. I.e. for a file with hourly time steps, -w 24 creates 24 hour accumulations. For leadtime 36 this is the sum of leadtimes 13-36.")
    parser.add_argument('ifile', help="Verif text or NetCDF file (input)")
    parser.add_argument('ofile', help="Verif NetCDF file (output)")
    parser.add_argument('-w', type=int, help="Accumulation window (in number of timesteps). If omitted, accumulate the whole leadtime axis.", dest="w")
    parser.add_argument('-i', help="Ignore missing values in the sum", action="store_true", dest="ignore")
    parser.add_argument('-x', default="leadtime", help="Axis", choices=["leadtime", "time"], dest="axis")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    ifile = verif.input.get_input(args.ifile)
    locations = ifile.locations
    locationids = [loc.id for loc in locations]
    leadtimes = ifile.leadtimes
    times = ifile.times
    lats = [loc.lat for loc in locations]
    lons = [loc.lon for loc in locations]
    elevs = [loc.elev for loc in locations]

    fcst = copy.deepcopy(ifile.fcst)
    obs = copy.deepcopy(ifile.obs)
    if args.axis == 'leadtime':
        axis = 1
    elif args.axis == 'time':
        axis = 0

    if args.w is None:
        if args.ignore:
            fcst = np.nancumsum(fcst, axis=axis)
            obs = np.nancumsum(obs, axis=axis)
        else:
            fcst = np.cumsum(fcst, axis=axis)
            obs = np.cumsum(obs, axis=axis)

    elif args.w > 1:
        # if args.w % 2 == 0:
        #     verif.util.error("Window length has to be an odd number")

        fcst = convolve(fcst, args.w, args.ignore, args.axis)
        obs = convolve(obs, args.w, args.ignore, args.axis)

    file = netCDF4.Dataset(args.ofile, 'w', format="NETCDF4")
    file.createDimension("leadtime", len(ifile.leadtimes))
    file.createDimension("time", None)
    file.createDimension("location", len(ifile.locations))
    vTime = file.createVariable("time", "i4", ("time",))
    vOffset = file.createVariable("leadtime", "f4", ("leadtime",))
    vLocation = file.createVariable("location", "f8", ("location",))
    vLat = file.createVariable("lat", "f4", ("location",))
    vLon = file.createVariable("lon", "f4", ("location",))
    vElev = file.createVariable("altitude", "f4", ("location",))
    vfcst = file.createVariable("fcst", "f4", ("time", "leadtime", "location"))
    vobs =  file.createVariable("obs", "f4", ("time", "leadtime", "location"))
    file.long_name = ifile.variable.name
    file.units = unit = ifile.variable.units.replace("$", "")
    file.Convensions = "verif_1.0.0"

    vobs[:] = obs
    vfcst[:] = fcst
    vTime[:] = times
    vOffset[:] = leadtimes
    vLocation[:] = locationids
    vLat[:] = lats
    vLon[:] = lons
    vElev[:] = elevs

    file.close()

if __name__ == '__main__':
    main()
