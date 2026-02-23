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
    parser = argparse.ArgumentParser(description='This program adds pre-computed ensemble statistics to a Verif file. It also renames variables using an older convention, if needed.')
    parser.add_argument('file', help='Input file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Make a copy, in case the append corrupts the file (shouldn't happen)
    filename_backup = args.file + ".tmp"
    shutil.copy(args.file, filename_backup)

    with netCDF4.Dataset(args.file, 'a') as file:
        # Rename variables
        if "crps" in file.variables:
            file.renameVariable("crps", "ensemble_crps")
        if "ens-mean" in file.variables:
            file.renameVariable("ens-mean", "ensemble_mean")
        if "ens-std" in file.variables:
            file.renameVariable("ens-std", "ensemble_std")
        if "ens-var" in file.variables:
            file.renameVariable("ens-var", "ensemble_variance")

        if "ensemble" not in file.variables:
            print(f"File does not have ensemble, cannot compute ensemble statistics.")
        else:
            ensemble = file.variables["ensemble"][:]
            num_members = ensemble.shape[-1]

            # Compute CRPS
            if "ensemble_crps" not in file.variables:
                file.createvariable("ensemble_crps", "f4", ("time", "leadtime", "location"))

            crps = compute_crps(ensemble, file.variables["obs"][:], num_members)
            num_valid_members = np.sum(~np.isnan(ensemble), axis=3)
            crps[num_valid_members != num_members] = np.nan
            file.variables["ensemble_crps"][:] = crps

            # Compute ensemble moments
            fields = ["ensemble_mean" , "ensemble_variance"]
            for field in fields:
                if field not in file.variables:
                    file.createVariable(field, "f4", ("time", "leadtime", "location"))

            ens_mean = np.nanmean(ensemble, axis=3)
            file.variables["ensemble_mean"][:] = ens_mean
            file.variables["ensemble_variance"][:] = np.nanvar(ensemble, axis=3)

    os.remove(filename_backup)


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
