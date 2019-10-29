import sys
import argparse
import numpy as np
import verif.input
import netCDF4
import copy
import astropy.convolution
import scipy.signal


def calculate_window(array, threshold, interval, leadtimes):
    """ Computes a weather window """
    O = array.shape[1]
    Inan = np.isnan(array)
    for o in range(0, O):
        q = np.nansum(interval.within(np.cumsum(array[:, o:, :], axis=1)), axis=1)
        I = q+o
        I[I >= O] = O-1
        array[:, o, :] = leadtimes[I] - leadtimes[o]
    array[Inan] = np.nan
    return array


def main():
    parser = argparse.ArgumentParser(prog="window", description="Converts observations and forecasts into weather windows: A window is the leng of time that a parameter is below a certain threshold, for example a dry-spell is the length (along the leadtime axis) that precipitation is 0 mm.")
    parser.add_argument('ifile', help="Verif text or NetCDF file (input)")
    parser.add_argument('ofile', help="Verif NetCDF file (output)")
    parser.add_argument('-r', type=float, help="Threshold for which the value must be above/below", dest="threshold", required=True)
    parser.add_argument('-b', default="below=", help="Bin type", dest="bin_type")

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

    intervals = verif.util.get_intervals(args.bin_type, [args.threshold])
    if len(intervals) != 1:
        verif.util.error("Improper bin type '%s'" % args.bin_type)
    interval = intervals[0]

    fcst = copy.deepcopy(ifile.fcst)
    obs = copy.deepcopy(ifile.obs)

    fcst = calculate_window(fcst, args.threshold, interval, leadtimes)
    obs = calculate_window(obs, args.threshold, interval, leadtimes)

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
    vobs = file.createVariable("obs", "f4", ("time", "leadtime", "location"))
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
