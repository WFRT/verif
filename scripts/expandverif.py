import argparse
import copy
import netCDF4
import numpy as np
import sys
import time as timing


import verif.util


removeMissing = True
nc_missing = netCDF4.default_fillvals["f4"]


def progress_bar(fraction, width, text=""):
    num_x = int(fraction * (width-len(text) - 2))
    num_space = width - num_x - len(text) - 2
    sys.stdout.write("\r" + text + "[" + "X" * num_x + " " * num_space + "]")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='Expands the observations in a verif file to match desired leadtimes and initialization times ')
    parser.add_argument('file', help='Verif file')
    parser.add_argument('-o', metavar="FILE", help='Output verif file', dest="ofilename", required=True)
    parser.add_argument('-i', type=verif.util.parse_numbers, default=[0], help='Initialization times in UTC (hours)', dest="init_times")
    parser.add_argument('-lt', type=verif.util.parse_numbers, help='Lead times (hours)', dest="lead_times")
    parser.add_argument('-t', type=verif.util.parse_numbers, help='Thresholds', dest="thresholds")
    parser.add_argument('-q', type=verif.util.parse_numbers, help='Quantiles', dest="quantiles")
    parser.add_argument('--debug', help='Display debug information', action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.debug:
        width = verif.util.get_screen_width()

    input = verif.input.get_input(args.file)
    itimes = input.times
    ileadtimes = input.leadtimes
    locations = input.locations
    variable = input.variable

    itimes_whole_days = np.unique(np.array([int(t / 86400) * 86400 for t in input.times]))
    otimes = np.zeros(len(itimes_whole_days) * len(args.init_times))
    for i in range(len(args.init_times)):
        I = range(i * len(itimes_whole_days), (i+1) * len(itimes_whole_days))
        otimes[I] = itimes_whole_days + args.init_times[i] * 3600
    oleadtimes = args.lead_times
    obs = nc_missing * np.ones([len(otimes), len(oleadtimes), len(locations)], float)
    shape = input.obs.shape
    allobs = np.reshape(input.obs, [shape[0]*shape[1], shape[2]])
    q, w = np.meshgrid(ileadtimes, itimes)
    alltimes = q * 3600 + w

    for t in range(len(otimes)):
        if args.debug and (len(otimes) < width or t % int(len(otimes)/width) == 0):
            progress_bar(t*1.0 / len(otimes), width)

        for lt in range(len(oleadtimes)):
            I = np.where(alltimes.flatten() == otimes[t] + oleadtimes[lt] * 3600)[0]
            if len(I) >= 1:
                obs[t, lt, :] = allobs[I[0], :]

    output = netCDF4.Dataset(args.ofilename, 'w')
    output.createDimension("time", None)
    output.createDimension("leadtime", len(oleadtimes))
    output.createDimension("location", len(locations))
    if args.thresholds is not None:
        output.createDimension("threshold", len(args.thresholds))
        var = output.createVariable("threshold", "f4", ("threshold"))
        var[:] = args.thresholds
        output.createVariable("cdf", "f4", ("time", "leadtime", "location", "threshold"))
    if args.quantiles is not None:
        output.createDimension("quantile", len(args.quantiles))
        var = output.createVariable("quantile", "f4", ("quantile"))
        var[:] = args.quantiles
        output.createVariable("x", "f4", ("time", "leadtime", "location", "quantile"))
    vTime = output.createVariable("time", "i4", ("time",))
    vOffset = output.createVariable("leadtime", "f4", ("leadtime",))
    vLocation = output.createVariable("location", "i4", ("location",))
    vLat = output.createVariable("lat", "f4", ("location",))
    vLon = output.createVariable("lon", "f4", ("location",))
    vElev = output.createVariable("altitude", "f4", ("location",))
    vfcst = output.createVariable("fcst", "f4", ("time", "leadtime", "location"))
    vobs = output.createVariable("obs", "f4", ("time", "leadtime", "location"))
    output.standard_name = variable.name
    output.units = unit = variable.units.replace("$", "")

    vobs[:] = obs
    vTime[:] = otimes
    vOffset[:] = oleadtimes
    vLocation[:] = [s.id for s in locations]
    vLat[:] = [s.lat for s in locations]
    vLon[:] = [s.lon for s in locations]
    vElev[:] = [s.elev for s in locations]
    output.Conventions = "verif_1.0.0"
    output.close()


if __name__ == '__main__':
    main()
