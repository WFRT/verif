import numpy as np
import verif.input
import verif.data
import verif.field
import netCDF4
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(prog="text2verif", description="Convert between Verif text and NetCDF files")
    parser.add_argument('ifile', type=str, help="Verif text file (input)")
    parser.add_argument('ofile', type=str, help="Verif NetCDF file (output)")
    parser.add_argument('--debug', help='Print debug information', action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    input = verif.input.get_input(args.ifile)
    times = input.times
    leadtimes = input.leadtimes
    locations = input.locations
    variable = input.variable

    output = netCDF4.Dataset(args.ofile, 'w')
    output.createDimension("time", None)
    output.createDimension("leadtime", len(leadtimes))
    output.createDimension("location", len(locations))
    thresholds = input.thresholds
    quantiles = input.quantiles
    if len(thresholds) > 0:
        if args.debug:
            print "Adding %d thresholds" % len(thresholds)
        output.createDimension("threshold", len(thresholds))
        output.createVariable("threshold", "f4", ["threshold"])
        output["threshold"][:] = thresholds
        output.createVariable("cdf", "f4", ("time", "leadtime", "location", "threshold"))
        output.variables["cdf"][:] = input.threshold_scores
    if len(quantiles) > 0:
        if args.debug:
            print "Adding %d quantiles" % len(quantiles)
        output.createDimension("quantile", len(quantiles))
        output.createVariable("quantile", "f4", ["quantile"])
        output["quantile"][:] = quantiles
        output.createVariable("x", "f4", ("time", "leadtime", "location", "quantile"))
        output.variables["x"][:] = input.quantile_scores

    vTime=output.createVariable("time", "i4", ("time",))
    vOffset=output.createVariable("leadtime", "f4", ("leadtime",))
    vLocation=output.createVariable("location", "i4", ("location",))
    vLat=output.createVariable("lat", "f4", ("location",))
    vLon=output.createVariable("lon", "f4", ("location",))
    vElev=output.createVariable("altitude", "f4", ("location",))
    vfcst=output.createVariable("fcst", "f4", ("time", "leadtime", "location"))
    vobs=output.createVariable("obs", "f4", ("time", "leadtime", "location"))

     # Create nonstandard fields
    standard = [verif.field.Obs(), verif.field.Fcst()]
    fields = [field for field in input.get_fields() if field not in standard]
    for field in fields:
        name = field.name()
        if field.__class__ == verif.field.Other:
            if args.debug:
                print "Adding non-standard score '%s'" % name
            output.createVariable(name, "f4", ("time", "leadtime", "location"))
            output.variables[name][:] = input.other_score(field.name())[:]

    output.standard_name = variable.name
    output.units = unit = variable.units.replace("$", "")

    vobs[:] = input.obs
    vfcst[:] = input.fcst
    vTime[:] = input.times
    vOffset[:] = input.leadtimes
    vLocation[:] = [s.id for s in locations]
    vLat[:] = [s.lat for s in locations]
    vLon[:] = [s.lon for s in locations]
    vElev[:] = [s.elev for s in locations]
    output.Conventions = "verif_1.0.0"
    output.close()


if __name__ == '__main__':
    main()
