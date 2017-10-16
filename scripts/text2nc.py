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
   vTime=output.createVariable("time", "i4", ("time",))
   vOffset=output.createVariable("leadtime", "f4", ("leadtime",))
   vLocation=output.createVariable("location", "i4", ("location",))
   vLat=output.createVariable("lat", "f4", ("location",))
   vLon=output.createVariable("lon", "f4", ("location",))
   vElev=output.createVariable("altitude", "f4", ("location",))
   vfcst=output.createVariable("fcst", "f4", ("time", "leadtime", "location"))
   vobs=output.createVariable("obs", "f4", ("time", "leadtime", "location"))
   output.standard_name = variable.name
   output.units = unit = variable.units

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
