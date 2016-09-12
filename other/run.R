# Example script using gamlss2verif
#
# This creates the fit.nc file. Afterwards, run verif like this:
# verif fit.nc -m reliability -r 0
# verif fit.nc -m pithist
# verif fit.nc -m mae -type map
source("gamlss2verif.R")

# Create random data
obs = rnorm(1000, 2, 5)
ensmean = obs+rnorm(1000,2,6)
ensspread = exp(rnorm(1000,2,1))
data = data.frame(LAT=60+rnorm(1000, 0, 0.3), LON=10+rnorm(1000, 0, 0.3), ELEV=5, SITE=1:1000,
                  DATE=20150101, OFFSET=(1:1000) %% 10,
                  OBS=obs, MEAN=ensmean, SPREAD=ensspread)
model = list(mu=OBS~MEAN, sigma=~SPREAD, family=NO)

# Crate a training dataset. Normally this could be done by selecting a certain date range for the
# training, and a different date range for the evaluation.
training = data[1:900,]

# Create an evaluation dataset. For this to work, the evaluation dataset must have offsets in common
# with the training dataset, otherwise it cannot train each offset separately.
evaluation = data[901:1000,]

gamlss2verif(model, training, evaluation, "fit.nc")
