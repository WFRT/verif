import argparse
import copy
import netCDF4
import numpy as np
import scipy
import sys


import verif.input


def main():
   parser = argparse.ArgumentParser(prog="ens2prob", description="Compute probabilistic information from ensembles")
   parser.add_argument('ifile', help="Verif text or NetCDF file (input)")
   parser.add_argument('ofile', help="Verif NetCDF file (output)")
   parser.add_argument('-r', default=[], type=verif.util.parse_numbers, help="Which thresholds (e.g.  temperatures) to compute CDF values for?", dest="thresholds")
   parser.add_argument('-q', default=[], type=verif.util.parse_numbers, help="Which quantiles (between 0 and 1) to compute?", dest="quantiles")
   parser.add_argument('-p', help="Compute PIT values?", dest="pit", action="store_true")

   if len(sys.argv) == 1:
      parser.print_help()
      sys.exit(0)

   args = parser.parse_args()

   ifile = verif.input.get_input(args.ifile)
   locations = ifile.locations
   locationids = [loc.id for loc in locations]
   leadtimes  = ifile.leadtimes
   times = ifile.times
   lats = [loc.lat for loc in locations]
   lons = [loc.lon for loc in locations]
   elevs = [loc.elev for loc in locations]

   """ Initialize """
   fcst = copy.deepcopy(ifile.fcst)
   obs = copy.deepcopy(ifile.obs)
   ens = copy.deepcopy(ifile.ensemble)
   ens = np.sort(ens, axis=3)
   if len(args.thresholds) > 0:
      cdf = np.zeros([obs.shape[0], obs.shape[1], obs.shape[2], len(args.thresholds)])
   if len(args.quantiles) > 0:
      x = np.zeros([obs.shape[0], obs.shape[1], obs.shape[2], len(args.quantiles)])
   if args.pit:
      pit = np.nan * np.zeros([obs.shape[0], obs.shape[1], obs.shape[2]])

   M = ens.shape[3]

   """
   Two approaches to assigning probabilities to ensemble members:
   1) Assign a cumulative probability of 0 to the lowest member and 1 to the highest
   2) Assign a cumulative probability of 1 / (M+1) to the lowest and M / (M+1) to the highest
   In either case, if a value is outside the ensemble range, set the value to the CDF assigned to
   the half of its nearest member.

   Then there are two approach for interpolating probabilities:
   a) round down to the nearest member
   b) linearly interpolate

   """
   lower_cdf = 0  # 1.0 / (M + 1)
   upper_cdf = 1 - lower_cdf

   """
   Compute cumulative probabilities at different thresholds

   Currently, only approach a) is supported.
   """
   for i, threshold in enumerate(args.thresholds):
      cdf[:, :, :, i] = np.mean(ens < threshold, axis=3) * (upper_cdf - lower_cdf) + lower_cdf / 2

   """
   Compute values for different quantiles

   Use kind='linear' for approach b)
   """
   for i, quantile in enumerate(args.quantiles):
      f = scipy.interpolate.interp1d(np.linspace(lower_cdf, upper_cdf, M),
              ens, bounds_error=False,
              axis=3, kind='zero')
      if quantile == 1:
          x[:, :, :, i] = ens[:, :, :, -1]
      else:
         x[:, :, :, i] = f(quantile)

   """
   Compute PIT values, i.e. the CDF at the observed value
   
   Currently, only approach a) is supported.
   """
   if args.pit:
      if obs is None:
         print("Error: File is missing obs, and can therefore not compute PIT")
         sys.exit(1)
      newobs = np.tile(np.expand_dims(obs, 3), [1, 1,1, M])
      pit = np.mean(ens < newobs, axis=3)
      """
      # approach b)
      for i in range(0, obs.shape[0]):
          for j in range(0, obs.shape[1]):
              for t in range(0, obs.shape[2]):
                  if np.isnan(obs[i, j , t]) == 0:
                      pit[i, j, t] = np.interp(obs[i, j, t], ens[i, j, t, :], np.linspace(lower_cdf, upper_cdf, M), left=lower_cdf/2, right=1-(1-upper_cdf)/2)
      """

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
   if fcst is not None:
       vFcst=file.createVariable("fcst", "f4", ("time", "leadtime", "location"))
   if obs is not None:
       vObs=file.createVariable("obs", "f4", ("time", "leadtime", "location"))
   if len(args.thresholds) > 0:
      vCdf=file.createVariable("cdf", "f4", ("time", "leadtime", "location", "threshold"))
      vThreshold=file.createVariable("threshold", "f4", ("threshold"))
   if len(args.quantiles) > 0:
      vX=file.createVariable("x", "f4", ("time", "leadtime", "location", "quantile"))
      vQuantile=file.createVariable("quantile", "f4", ("quantile"))
   if args.pit:
      vPit=file.createVariable("pit", "f4", ("time", "leadtime", "location"))
   file.Variable = ifile.variable.name
   file.units = unit = ifile.variable.units.replace("$", "")
   file.Convensions = "verif_1.0.0"
   new_history = ' '.join(sys.argv)
   if hasattr(file, 'history'):
       file.history = file.history + '\n' + new_history
   else:
       file.history = new_history

   if obs is not None:
      vObs[:] = obs
   if fcst is not None:
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
   if args.pit:
      vPit[:] = pit

   file.close()

if __name__ == '__main__':
   main()
