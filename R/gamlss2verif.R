library(ncdf4)
require(gamlss)

# Creates a verification file that can be read by verif (https://github.com/wfrt/verif)
# by fitting a gamlss model for each leadtime.
#
# Arguments:
# model: A named list containing the specification of a gamlss model, where the predictors must be
#    columns in the dataframes provided:
#    mu: a formula for mu, e.g. mu=obs~ens_mean + x + y
#    sigma: a formula for sigma, e.g. sigma=ens_spread
#    nu: a formula for nu
#    tau: a formula for nu
#    family: A gamlss model, e.g. NO, ZAGA, BCT, etc
#    nu and tau may be omitted if the family does not require it
#
#    Use "raw" instead of a list if you just want the raw forecast and "clim" to create a
#    file based on climatolocical observations.
# xtrain: dataframe containing training data with the following columns:
#    location: Location Id
#    lat: Location latitude (degrees)
#    lon: Location longitude (degrees)
#    altitude: Location elevation (meters)
#    time: Forecast initialization time (unix-time)
#    leadtime: Forecast leadtime (hours)
#    obs: Observation
#    Whatever columns are needed by 'model'
#    ens1...ensN: (Optional) Put each ensemble member in a separate column, needed when model="raw"
# xeval: evaluation dataset with the same format as 'x'
# filename: where should the verification data be stored?
# name: Name of the forecast variable (e.g. Precip,  T, or WindSpeed).
# units: Units of the variable (e.g. mm, ^oC, or m/s)
# quantiles: Which quantiles should be written?
# thresholds: Which thresholds should probabilities be computed for?
# leadtimeRange: Allow data from neighbouring leadtimes to create the training. Makes the calibration
#            more robust at the expense (benefit?) of smoothening the coefficient across leadtimes.
#            A value of 1 means use data with leadtimes +-1.
# useMedian: Should the median be used to create the deterministic forecast? Otherwise the mean
#            but this might not work for anything other than NO and ZAGA.
# debug: If TRUE, show debug information
gamlss2verif <- function(model, xtrain, xeval, filename, name=NULL, units=NULL,
                    quantiles=c(0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99),
                    thresholds=NULL,
                    leadtimeRange=0,
                    useMedian=TRUE,
                    debug=FALSE) {
   # Missing value indicator for NetCDF
   MV = 9.96921e+36

   # Dimensions
   times      <- sort(unique(xeval$time))
   leadtimes    <- intersect(sort(unique(xeval$leadtime)), sort(unique(xtrain$leadtime)))
   locations  <- sort(unique(xeval$location))
   dTime     <- ncdim_def("time", "", times)
   dLeadtime   <- ncdim_def("leadtime", "", leadtimes)
   dLocation <- ncdim_def("location", "", locations)
   if(length(thresholds) > 0)
      dThreshold <- ncdim_def("threshold", "", thresholds)
   if(length(quantiles) > 0)
      dQuantile <- ncdim_def("quantile", "", quantiles)

   # Remove missing stations (i.e. stations that are only in training set)
   I = which(xeval$location %in% locations)
   xeval = xeval[I,]

   # Variables
   Iloc <- match(locations, xeval$location)
   vLat <- ncvar_def("lat", "degrees", dLocation, NULL)
   vLon <- ncvar_def("lon", "degrees", dLocation, NULL)
   vElev <- ncvar_def("altitude", "m", dLocation, NULL)
   vObs  <- ncvar_def("obs", "", list(dLocation, dLeadtime, dTime), NULL)
   vFcst <- ncvar_def("fcst", "", list(dLocation, dLeadtime, dTime), NULL)
   vPit  <- ncvar_def("pit", "", list(dLocation, dLeadtime, dTime), NULL)
   vIgn  <- ncvar_def("ign", "", list(dLocation, dLeadtime, dTime), NULL)
   varList <- list(vLat, vLon, vElev, vObs, vFcst, vPit, vIgn)
   if(length(thresholds) > 0) {
      vThreshold <- ncvar_def("threshold", "", list(dThreshold), NULL)
      vCdf <- ncvar_def("cdf", "", list(dThreshold, dLocation, dLeadtime, dTime), NULL)
      varList <- c(varList, list(vCdf))
   }
   if(length(quantiles) > 0) {
      vQuantile <- ncvar_def("quantile", "", list(dQuantile), NULL)
      vX <- ncvar_def("x", "", list(dQuantile, dLocation, dLeadtime, dTime), NULL)
      varList <- c(varList, list(vX))
   }

   fid <- nc_create(filename, varList)
   nc_close(fid)

   # Set up data

   fid <- nc_open(filename, write=TRUE)
   ncvar_put(fid, vLat, xeval$lat[Iloc])
   ncvar_put(fid, vLon, xeval$lon[Iloc])
   ncvar_put(fid, vElev, xeval$altitude[Iloc])

   # Compute scores
   xfcst <- array(MV, dim(xeval)[1])
   xpit  <- array(MV, dim(xeval)[1])
   xign  <- array(MV, dim(xeval)[1])
   if(length(thresholds) > 0)
      xp    <- array(0, c(length(xfcst), length(thresholds)))
   if(length(quantiles) > 0)
      xq    <- array(0, c(length(xfcst), length(quantiles)))
   for(i in 1:length(leadtimes)) {
      lt = leadtimes[i]
      if(debug)
         print(paste("Leadtime:", lt))
      I = which(abs(xtrain$leadtime - lt) <= leadtimeRange)
      xt = xtrain[I,]
      I = which(xeval$leadtime == lt)
      xe = xeval[I,]
      if(is.character(model)) {
         fit = model
      }
      else {
         mu=model$mu
         sigma=model$sigma
         nu=model$nu
         tau=model$tau
         family=model$family
         fit = gamlss(mu, sigma.formula=sigma, nu.formula=nu, tau.formula=tau, family=family, data=xt)
         if(debug)
            print(fit)
      }

      # Precompute parameters
      par <- getMoments(fit, xt, xe)

      if(useMedian)
         xfcst[I] <- qG(0.5, fit, xe, par)
      else
         xfcst[I] <- mG(fit, xe, par)  # Mean, doesn't seem to work for many distributions
      xpit[I]  <- pG(xe$obs, fit, xe, par)
      # We don't need to randomize PIT, because verif does that
      xign[I]  <- -log2(dG(xe$obs, fit, xe, par))
      if(length(thresholds) > 0) {
         for(c in 1:length(thresholds)) {
            xp[I,c] <- pG(thresholds[c], fit, xe, par)
         }
      }
      if(length(quantiles) > 0) {
         for(c in 1:length(quantiles)) {
            xq[I,c] <- qG(quantiles[c], fit, xe, par)
         }
      }
   }

   obs   <- array(MV, c(length(locations), length(leadtimes), length(times)))
   fcst  <- array(MV, c(length(locations), length(leadtimes), length(times)))
   pit   <- array(MV, c(length(locations), length(leadtimes), length(times)))
   ign   <- array(MV, c(length(locations), length(leadtimes), length(times)))
   p     <- array(MV, c(length(thresholds), length(locations), length(leadtimes), length(times)))
   q     <- array(MV, c(length(quantiles), length(locations), length(leadtimes), length(times)))
   for(d in 1:length(times)) {
      Id = which(xeval$time == times[d])
      for(o in 1:length(leadtimes)) {
         Io = which(xeval$leadtime == leadtimes[o])
         I0 = intersect(Id, Io)
         I  = match(xeval$location[I0], locations)
         obs[I,o,d]  = xeval$obs[I0]
         fcst[I,o,d] = xfcst[I0]
         pit[I,o,d]  = xpit[I0]
         ign[I,o,d]  = xign[I0]
         if(length(thresholds) > 0) {
            for(c in 1:length(thresholds)) {
               p[c,I,o,d] = xp[I0,c]
            }
         }
         if(length(quantiles) > 0) {
            for(c in 1:length(quantiles)) {
               q[c,I,o,d] = xq[I0,c]
            }
         }
      }
   }
   ncvar_put(fid, vObs, obs)
   ncvar_put(fid, vFcst, fcst)
   ncvar_put(fid, vPit, pit)
   if(length(which(is.na(ign))) == 0 && length(which(is.infinite(ign))) == 0)
      ncvar_put(fid, vIgn, ign)

   if(!is.null(name))
      ncatt_put(fid, 0, "long_name", name)
   if(!is.null(units))
      ncatt_put(fid, 0, "units", units)
   ncatt_put(fid, 0, "verif_version", "verif_1.0.0")

   if(length(thresholds) > 0) {
      ncvar_put(fid, vCdf, p)
   }
   if(length(quantiles) > 0) {
      ncvar_put(fid, vX, q)
   }
   nc_close(fid)
}

pG <- function(p, fit, x, par=NULL) {
   if(length(p) == 1)
      p = 0*x$obs + p
   return(getValues(p, fit, x, "p", par))
}
qG <- function(q, fit, x, par=NULL) {
   if(length(q) == 1)
      q = 0*x$obs + q
   values = getValues(q, fit, x, "q", par)
   return(values)
}
dG <- function(d, fit, x, par=NULL) {
   if(length(d) == 1)
      d = 0*x$obs + d
   return(getValues(d, fit, x, "d", par))
}
mG <- function(fit, x, par=NULL) {
   return(getValues(0, fit, x, "m", par))
}
# Compute moments for 'x'. For some reason, gamlss's predictAll requires access to the
# training data used to create 'fit'. This is passed in as 'xfit'.
getMoments <- function(fit, xfit, x) {
   cls = class(fit)
   if(length(cls) != 4 || cls[1] != "gamlss") {
      kens = grep("ens", names(x))
      par  = t(apply(x[,kens], 1, sort))
      return(par)
   }
   family = fit$family[1]
   par = predictAll(fit, data=xfit, newdata=x)
   return(par)
}
getValues <- function(q, fit, x, type, par=NULL) {
   if(length(q) != dim(x)[1]) {
      #stop("getValues: q must be the same size as x")
   }
   if(is.character(fit)) {
      if(fit == "raw") {
         values = matrix(NA, nrow=dim(x)[1], ncol=1)
         kens = grep("ens", names(x))
         if(length(kens) == 0) {
            stop("No ensemble members (columns such as ens1) found in data frame")
         }
         if(type == "p") {
            values = apply(x[,kens] <= q, 1, mean)
         }
         else if(type == "q") {
            if(is.null(par)) {
               par  = t(apply(x[,kens], 1, sort))
            }
            Nrows = dim(par)[1]
            Nens  = dim(par)[2]
            k     = floor(Nens*q)
            k[which(k == 0)] = 1
            ind   = (k-1)*Nrows + (1:Nrows)
            values[,1] = par[ind]
         }
         else if(type == "d") {
            warning("PDF not implemented for raw")
            values[,1] = -999
         }
         else if(type == "m") {
            values[,1] = apply(x[,kens], 1, mean)
         }
      }
      else if(fit == "clim") {
         values = matrix(NA, nrow=dim(x)[1], ncol=1)
         if(type == "p") {
            sites = unique(x$location)
            for(site in sites) {
               I = which(x$location == site)
               values[I,1] = mean(x$obs[I] <= q)
            }
         }
         else if(type == "q") {
            sites = unique(x$location)
            for(site in sites) {
               I = which(x$location == site)
               if(length(I) > 0) {
                  temp  = sort(x$obs[I])
                  N     = length(temp)
                  ind   = floor(N*q[I])
                  ind[ind == 0] = 1
                  values[I,1] = temp[ind]
               }
            }
         }
         else if(type == "d") {
            warning("PDF not implemented for clim")
            values[,1] = -999
         }
         else if(type == "m") {
            sites = unique(x$location)
            for(site in sites) {
               I = which(x$location == site)
               values[I,1] = mean(x$obs[I])
            }
         }
      }
      else {
         stop(paste("Unrecognized model", model))
      }
   }
   else {
      family = fit$family[1]
      if(is.null(par)) {
         par = predictAll(fit, data=x, newdata=x)
      }
      mu = par$mu
      sigma = par$sigma
      nu = par$nu
      tau = par$tau

      # Computing the mean of the distribution:
      # For ZAGA the mean is mu*(1-nu)
      # For NO the mean is mu
      # For other distributions I am unsure...
      if(type == "m") {
         values = matrix(NA, nrow=dim(x)[1], ncol=1)
         if(family == "ZAGA" & !is.null(nu))
            values[,1] = mu*(1-nu)
         else
            values[,1] = mu
         if(family != "ZAGA" & family != "NO") {
            print(paste("Computing the mean of the distribution for", family, "is not tested..."))
         }
         return(values)
      }
      # qZAGA is really slow, use optimized function provided below
      if(type == "q" & family == "ZAGA") {
         string = "qzaga"
      }
      else {
         string = paste(type, family,sep="")
      }
      string = paste(string, "(q, mu=mu, sigma=sigma", sep="")
      if(!is.null(nu)) {
         string = paste(string, ",nu=nu", sep="")
      }
      if(!is.null(tau))
         string = paste(string, ",tau=tau", sep="")
      string = paste(string, ")", sep="")
      values <- eval(parse(text=string))
   }
   return(values)
}

# A faster implementation of qZAGA, by uing qgamma, instead of doing
# root finding with pZAGA.
qzaga <- function(p, mu, sigma, nu) {
   values <- array(length(p))
   I0 = nu > p
   values[I0] = 0
   I1 = nu <= p
   padj = (p[I1] - nu[I1]) /(1-nu[I1])

   shape <- 1 / (sigma[I1]*sigma[I1])
   scale <- sigma[I1]*sigma[I1]*mu[I1]

   values[I1] = qgamma(padj, shape=shape, scale=scale)

   return(values)
}
