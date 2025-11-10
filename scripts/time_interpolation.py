import os
import sys
import numpy as np
import matplotlib.pylab as mpl
import argparse
import verif.util
import netCDF4
import shutil
import scipy.interpolate


def main():
    """
    It takes an existing Verif file, extracts 6h forecasts, and fills in the intermediate hours with either
    linearly or splin interpolated values.
    """
    parser = argparse.ArgumentParser(description='This program downsamples a verif file and interpolates the data back using an interpolation method')
    parser.add_argument('ifile', help='')
    parser.add_argument('ofile', help='')
    parser.add_argument('-i', choices=['linear', 'spline'], dest='interpolator', required=True)
    parser.add_argument('-f', type=int, dest="frequency", required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    frequency = args.frequency

    if frequency < 2:
        raise ValueError("frequency must be > 1")

    shutil.copy(args.ifile, args.ofile)

    with netCDF4.Dataset(args.ofile, 'a') as file:
        leadtimes = file.variables["leadtime"][:]
        num_leadtimes = len(leadtimes)

        has_ens = "ensemble" in file.variables
        if has_ens:
            ensemble = file.variables["ensemble"][:]

        if frequency >= num_leadtimes:
            raise ValueError(f"frequency must be < {num_leadtimes}")

        fcst = file.variables["fcst"][:]

        if args.interpolator == "linear":
            ends = range(frequency, num_leadtimes+1, frequency)
            for end in ends:
                start = end - frequency

                fcst[:, (start+1):end, :] = np.nan
                fcst_start = fcst[:, start, :]
                fcst_end = fcst[:, end, :]
                for i in range(1, frequency):
                    w1 = i / frequency
                    w0 = 1 - w1
                    fcst[:, start + i, :] = w0 * fcst_start + w1 * fcst_end
                    # print(start, end, i, w0, w1)

                if has_ens:
                    ensemble[:, (start+1):end, :, :] = np.nan
                    fcst_start = ensemble[:, start, :, :]
                    fcst_end = ensemble[:, end, :]
                    for i in range(1, frequency):
                        w1 = i / frequency
                        w0 = 1 - w1
                        ensemble[:, start + i, :, :] = w0 * fcst_start + w1 * fcst_end

            # Trim the last leadtimes that cannot be interpolated
            fcst[:, ends[-1]+1:, :] = np.nan
            if has_ens:
                ensemble[:, ends[-1]+1:, :, :] = np.nan

        elif args.interpolator == "spline":
            y = fcst[:, 0::frequency, :]
            x = leadtimes[0::frequency]

            interpolator = scipy.interpolate.CubicSpline(x, y, axis=1)
            xx = set([i for i in leadtimes]) - set([i for i in x])
            xx = np.array(list(xx))
            temp = interpolator(xx)
            for i, index in enumerate(xx):
                fcst[:, int(index), :] = temp[:, i, :]

            if has_ens:
                y = ensemble[:, 0::frequency, :, :]
                x = leadtimes[0::frequency]

                interpolator = scipy.interpolate.CubicSpline(x, y, axis=1)
                xx = set([i for i in leadtimes]) - set([i for i in x])
                xx = np.array(list(xx))
                temp = interpolator(xx)
                for i, index in enumerate(xx):
                    ensemble[:, int(index), :, :] = temp[:, i, :, :]

        file.variables["fcst"][:] = fcst
        if has_ens:
            file.variables["ensemble"][:] = ensemble

        # These variables are no longer valid, since we cannot derive them for interpolated
        # timeseries
        invalid_variables = ["x", "cdf", "pdf", "pit"]
        for variable in invalid_variables:
            if variable in file.variables:
                file.variables[variable] = np.nan

        # Compute CRPS
        if has_ens:
            if not "crps" in file.variables:
                file.createVariable("crps", "f4", ("time", "leadtime", "location"))

            num_members = ensemble.shape[-1]
            crps = compute_crps(ensemble, file.variables["obs"][:], num_members)

            num_valid_members = np.sum(~np.isnan(ensemble), axis=3)
            crps[num_valid_members != num_members] = np.nan

            file.variables["crps"][:] = crps


def compute_crps(preds, targets, num_members, fair=True) -> np.ndarray:
    """Continuous Ranked Probability Score (CRPS).

    Taken from bris-inference

    Args:
        preds: numpy.ndarray
            Predictions, shape (time, leadtime, location, ens_size)
        targets: numpy.ndarray
            Targets, shape (time, leadtime, location)
        fair: bool
            Defaults to true

    Returns:
        crps: numpy.ndarray
            Shape (time, leadtime, location)
    """

    coef = (
        -1.0 / (num_members * (num_members - 1))
        if fair
        else -1.0 / (num_members**2)
    )

    mae = np.mean(np.abs(targets[..., None] - preds), axis=-1)

    # var = np.abs(preds[..., None] - preds[..., None, :])
    var = np.zeros(preds.shape[:-1])
    for i in range(num_members):  # loop version to reduce memory usage
        var += np.sum(np.abs(preds[..., i, None] - preds[..., i + 1 :]), axis=-1)
    var *= coef
    return mae + var


if __name__ == "__main__":
    main()
