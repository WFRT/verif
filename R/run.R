# Example script using gamlss2verif
#
# This creates the fit.nc file. Afterwards, run verif like this:
# verif fit.nc -m obsfcst -x month
# verif fit.nc -m reliability -r 0
# verif fit.nc -m pithist
source("gamlss2verif.R")

# Create random data
obs = rnorm(365*10, 2, 10) - cos(rep(1:365, each=10)/365*2*pi)*10
# To simulate forecasts, add noice to obs (grater noise for longer leadtimes)
ensmean = obs+rnorm(365*10,2,rep(seq(2, 6, length=10), 365))
# Simulate some spread/skill relationship
ensspread = abs(obs - ensmean) * exp(rnorm(365*10,2,1))
start_date = 20150101
end_date = 20151231
start_unixtime = as.numeric(as.POSIXct(as.character(start_date), format="%Y%m%d", origin=19700101))
end_unixtime = as.numeric(as.POSIXct(as.character(end_date), format="%Y%m%d", origin=19700101))
unixtimes = rep(seq(start_unixtime, end_unixtime, by=86400), each=10)
Ndays = length(unixtimes)
data = data.frame(LAT=59.9423, LON=10.72, ELEV=94, ID=18700,
                  TIME=unixtimes, LEADTIME=rep(1:10, Ndays),
                  OBS=obs, MEAN=ensmean, SPREAD=ensspread)
model = list(mu=OBS~MEAN, sigma=~SPREAD, family=NO)

# Crate a training dataset. Normally this could be done by selecting a certain date range for the
# training, and a different date range for the evaluation.
training = data[1:1800,]

# Create an evaluation dataset. For this to work, the evaluation dataset must have offsets in common
# with the training dataset, otherwise it cannot train each offset separately.
evaluation = data[1801:dim(data)[1],]

gamlss2verif(model, training, evaluation, "fit.nc", thresholds=c(0,5,10), name="Temperature", units="^oC")
