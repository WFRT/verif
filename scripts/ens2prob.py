import sys
import argparse
import numpy as np
import verif.input
import netCDF4
import copy
import astropy.convolution
import scipy
import scipy.signal


""" Convolves the Verif array across the leadtime dimension """
def convolve(array, window, ignore_missing):
   assert(len(array.shape) == 3)
   c = np.ones([1, window, 1])
   if ignore_missing:
      # Does this work?
      array[np.isnan(array)] = 0
   new_array = np.nan*np.zeros(array.shape)
   new_array[:, (window-1):, :] = scipy.signal.convolve(array, c, "valid")
   return new_array


def main():
   parser = argparse.ArgumentParser(prog="ens2prob", description="Converts ensemble information to probabilistic information")
   parser.add_argument('ifile', help="Verif text or NetCDF file (input)")
   parser.add_argument('ofile', help="Verif NetCDF file (output)")
   parser.add_argument('-r', type=verif.util.parse_numbers, help="Which thresholds to compute CDF values for?", dest="thresholds")
   parser.add_argument('-q', type=verif.util.parse_numbers, help="Which quantiles to compute values for?", dest="quantiles")

   if len(sys.argv) == 1:
      parser.print_help()
      sys.exit(0)

   args = parser.parse_args()

   ifile = verif.input.get_input(args.ifile)
   locations = ifile.locations
   locationids = [loc.id for loc in locations]
   leadtimes  = ifile.leadtimes
   times    = ifile.times
   lats     = [loc.lat for loc in locations]
   lons     = [loc.lon for loc in locations]
   elevs    = [loc.elev for loc in locations]

   fcst = copy.deepcopy(ifile.fcst)
   obs = copy.deepcopy(ifile.obs)
   ens = copy.deepcopy(ifile.ensemble)
   if len(args.thresholds) > 0:
      cdf = np.zeros([fcst.shape[0], fcst.shape[1], fcst.shape[2], len(args.thresholds)])
   if len(args.quantiles) > 0:
      x = np.zeros([fcst.shape[0], fcst.shape[1], fcst.shape[2], len(args.quantiles)])

   for i, threshold in enumerate(args.thresholds):
      cdf[:, :, :, i] = np.nanmean(ens < threshold, axis=3)

   M = ens.shape[3]
   """
   To compute inverse CDFs, assign quantiles to each ensemble member. There are several approaches:
   - Set the lowest member to  and the highest to 1
   - Set the lowest to 1 / (M+1) and the highest to M / (M+1)
   In either case, if a value is outside the ensemble range, set the value to the CDF assigned to
   the nearest member.
   """
   lower_cdf = 0  # 1.0 / (M + 1)
   upper_cdf = 1  # float(M) / (M + 1)
   for i, quantile in enumerate(args.quantiles):
      f = scipy.interpolate.interp1d(np.linspace(lower_cdf, upper_cdf, M), ens, bounds_error=False, fill_value=[lower_cdf, upper_cdf], axis=3)
      x[:, :, :, i] = f(quantile)

   file = netCDF4.Dataset(args.ofile, 'w', format="NETCDF4")
   file.createDimension("leadtime", len(ifile.leadtimes))
   file.createDimension("time", None)
   file.createDimension("location", len(ifile.locations))
   if len(args.thresholds) > 0:
      file.createDimension("threshold", len(args.thresholds))
   if len(args.quantiles) > 0:
      file.createDimension("quantile", len(args.quantiles))
   vTime=file.createVariable("time", "i4", ("time",))
   vOffset=file.createVariable("leadtime", "f4", ("leadtime",))
   vLocation=file.createVariable("location", "f8", ("location",))
   vLat=file.createVariable("lat", "f4", ("location",))
   vLon=file.createVariable("lon", "f4", ("location",))
   vElev=file.createVariable("altitude", "f4", ("location",))
   vFcst=file.createVariable("fcst", "f4", ("time", "leadtime", "location"))
   vObs=file.createVariable("obs", "f4", ("time", "leadtime", "location"))
   if len(args.thresholds) > 0:
      vCdf=file.createVariable("cdf", "f4", ("time", "leadtime", "location", "threshold"))
      vThreshold=file.createVariable("threshold", "f4", ("threshold"))
   if len(args.quantiles) > 0:
      vX=file.createVariable("x", "f4", ("time", "leadtime", "location", "quantile"))
      vQuantile=file.createVariable("quantile", "f4", ("quantile"))
   file.Variable = ifile.variable.name
   file.units = unit = ifile.variable.units.replace("$", "")
   file.Convensions = "verif_1.0.0"

   vObs[:] = obs
   vFcst[:] = fcst
   vTime[:] = times
   vOffset[:] = leadtimes
   vLocation[:] = locationids
   vLat[:] = lats
   vLon[:] = lons
   vElev[:] = elevs
   if len(args.thresholds) > 0:
      vThreshold[:] = args.thresholds
      vCdf[:] = cdf
   if len(args.quantiles) > 0:
      vQuantile[:] = args.quantiles
      vX[:] = x

   file.close()

if __name__ == '__main__':
   main()
