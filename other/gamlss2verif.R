library(ncdf4)
require(gamlss)

# Creates a verification file that can be read by verif (https://github.com/wfrt/verif)
# by fitting a gamlss model for each leadtime.
# model: A named list containing the specification of a gamlss model:
#    mu: a formula for mu, e.g. mu=OBS~MEAN
#    sigma: a formula for sigma, e.g. sigma=MEAN
#    nu: a formula for nu
#    tau: a formula for nu
#    family: A gamlss model, e.g. NO, ZAGA, BCT, etc
#    nu and tau may be omitted if the family does not require it
#    Use "raw" instead of a list if you just want the raw forecast.
# x: dataframe containing training data with the following columns:
#    SITE: Station Id
#    LAT: Station latitude
#    LON: Station longitude
#    ELEV: Station elevation (meters)
#    OFFSET: Forecast leadtime (hours)
#    DATE: Forecast initialization date (YYYYMMDD)
#    OBS: Observation
#    Whatever columns are needed by 'model'
#    ENS1...ENSN: (Optional) Put each ensemble member in a separate column, needed when model="raw"
# y: evaluation dataset with the same format as 'x'
# filename: where should the verification data be stored?
# variable: Which variable is this (Precip,  T, or WindSpeed). Used to write units, etc
# quantiles: Which quantiles should be written?
# thresholds: Which thresholds should probabilities be computed for?
# useMedian: Should the median be used to create the deterministic forecast? Otherwise the mean
#            but this might not work for anything other than NO and ZAGA.
#
# Example:
# obs = rnorm(1000, 2, 5)
# ensmean = obs+rnorm(1000,2,6)
# ensspread = exp(rnorm(1000,2,1))
# data = data.frame(LAT=60+rnorm(1000, 0, 0.3), LON=10+rnorm(1000, 0, 0.3), ELEV=5, SITE=1:1000,
#                   DATE=20150101, OFFSET=5,
#                   OBS=obs, MEAN=ensmean, SPREAD=ensspread)
# model = list(mu=OBS~MEAN, sigma=~SPREAD, family=NO)
# gamlss2verif(model, data, data, "fit.nc")
#
# Then run verif like this:
# verif fit.nc -m reliability -r 0
# verif fit.nc -m pithist
# verif fit.nc -m mae -type map

gamlss2verif <- function(model, x, xeval, filename, variable="Precip",
                    quantiles=c(0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99),
                    thresholds=NA,
                    useMedian=TRUE) {
   library("ncdf")
   MV  = -1e30
   MVL = -999
   # Which thresholds should CDFs be written for?
   if(is.na(thresholds)) {
      if(variable == "Precip" | variable == "WindSpeed") {
         thresholds <- c(0,0.2,0.3,0.4,0.5,0.7,1,2,5,10,15,20)
      }
      else if(variable == "T"){
         thresholds <- c(-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20)
      }
   }

   # Dimensions
   date      <- sort(unique(xeval$DATE))
   offset    <- sort(unique(xeval$OFFSET))
   offset    <- intersect(sort(unique(xeval$OFFSET)), sort(unique(x$OFFSET)))
   location  <- intersect(sort(unique(xeval$SITE)), sort(unique(x$SITE)))
   dDate     <- dim.def.ncdf("Date", "", date)
   dOffset   <- dim.def.ncdf("Offset", "", offset)
   dLocation <- dim.def.ncdf("Location", "", location)

   # Remove missing stations
   I = which(xeval$SITE %in% location)
   xeval = xeval[I,]

   # Variables
   Iloc <- match(location, xeval$SITE)
   vLat <- var.def.ncdf("Lat", "degrees", dLocation, MV)
   vLon <- var.def.ncdf("Lon", "degrees", dLocation, MV)
   vElev <- var.def.ncdf("Elev", "m", dLocation, MV)

   vObs  <- var.def.ncdf("obs", "", list(dLocation, dOffset, dDate), MV)
   vFcst <- var.def.ncdf("fcst", "", list(dLocation, dOffset, dDate), MV)
   vPit  <- var.def.ncdf("pit", "", list(dLocation, dOffset, dDate), MV)
   vIgn  <- var.def.ncdf("ign", "", list(dLocation, dOffset, dDate), MV)
   vSpread <- var.def.ncdf("spread", "", list(dLocation, dOffset, dDate), MV)

   varList <- list(vLat, vLon, vElev, vObs, vFcst, vPit, vIgn, vSpread)
   vcdf    <- NULL
   for(c in 1:length(thresholds)) {
      varname = getPVarName(thresholds[c])
      v <- var.def.ncdf(varname, "", list(dLocation, dOffset, dDate), MV)
      varList[length(varList)+1] <- list(v)
   }
   for(c in 1:length(quantiles)) {
      varname = getQVarName(quantiles[c])
      v <- var.def.ncdf(varname, "", list(dLocation, dOffset, dDate), MV)
      varList[length(varList)+1] <- list(v)
   }

   fid <- create.ncdf(filename, varList)
   close.ncdf(fid)

   # Set up data

   fid <- open.ncdf(filename, write=TRUE)
   put.var.ncdf(fid, vLat, xeval$LAT[Iloc])
   put.var.ncdf(fid, vLon, xeval$LON[Iloc])
   put.var.ncdf(fid, vElev, xeval$ELEV[Iloc])

   # Compute scores
   xfcst <- array(-999, dim(xeval)[1])
   xpit  <- array(-999, dim(xeval)[1])
   xspread  <- array(-999, dim(xeval)[1])
   xign  <- array(-999, dim(xeval)[1])
   xp    <- array(0, c(length(xfcst), length(thresholds)))
   xq    <- array(0, c(length(xfcst), length(quantiles)))
   for(i in 1:length(offset)) {
      off = offset[i]
      print(paste("Offset:", off))
      I = which(x$OFFSET == off)
      xt = x[I,]
      I = which(xeval$OFFSET == off)
      xe = xeval[I,]
      if(!identical(model, "raw")) {
         mu=model$mu
         sigma=model$sigma
         nu=model$nu
         tau=model$tau
         family=model$family
         fit = gamlss(mu, sigma.formula=sigma, nu.formula=nu, tau.formula=tau, family=family, data=xt)
      }
      else {
         fit = model
      }

      # Precompute parameters
      par <- getMoments(fit, xe)

      if(useMedian)
         xfcst[I] <- qG(0.5, fit, xe, par)
      else
         xfcst[I] <- mG(fit, xe, par)  # Mean, doesn't seem to work for many distributions
      xpit[I]  <- pG(xe$OBS, fit, xe, par)
      xspread[I]  <- (qG(0.84, fit, xe, par)-qG(0.16, fit, xe, par))/2
      # We don't need to randomize PIT, because verif does that
      xign[I]  <- -log2(dG(xe$OBS, fit, xe, par))
      for(c in 1:length(thresholds)) {
         xp[I,c] <- pG(thresholds[c], fit, xe, par)
      }
      for(c in 1:length(quantiles)) {
         xq[I,c] <- qG(quantiles[c], fit, xe, par)
      }
   }

   #stopifnot(length(xfcst) == length(x$OBS))
   obs   <- array(MVL, c(length(location), length(offset), length(date)))
   fcst  <- array(MVL, c(length(location), length(offset), length(date)))
   pit   <- array(MVL, c(length(location), length(offset), length(date)))
   spread <- array(MVL, c(length(location), length(offset), length(date)))
   ign   <- array(MVL, c(length(location), length(offset), length(date)))
   p     <- array(MVL, c(length(location), length(offset), length(date),
                         length(thresholds)))
   q     <- array(MVL, c(length(location), length(offset), length(date), length(quantiles)))
   for(d in 1:length(date)) {
      # print(paste(d, "/", length(date), sep=""))
      Id = which(xeval$DATE == date[d])
      for(o in 1:length(offset)) {
         Io = which(xeval$OFFSET == offset[o])
         I0 = intersect(Id, Io)
         I  = match(xeval$SITE[I0], location)
         obs[I,o,d]  = xeval$OBS[I0]
         fcst[I,o,d] = xfcst[I0]
         pit[I,o,d]  = xpit[I0]
         spread[I,o,d]  = xspread[I0]
         ign[I,o,d]  = xign[I0]
         for(c in 1:length(thresholds)) {
            p[I,o,d,c] = xp[I0,c]
         }
         for(c in 1:length(quantiles)) {
            q[I,o,d,c] = xq[I0,c]
         }
      }
   }
   put.var.ncdf(fid, vObs, obs)
   put.var.ncdf(fid, vFcst, fcst)
   put.var.ncdf(fid, vPit, pit)
   put.var.ncdf(fid, vSpread, spread)
   if(length(which(is.na(ign))) == 0 && length(which(is.infinite(ign))) == 0)
      put.var.ncdf(fid, vIgn, ign)

   att.put.ncdf(fid, 0, "Variable", variable)
   if(variable == "Precip")
      att.put.ncdf(fid, 0, "Units", "mm")
   else if(variable == "WindSpeed")
      att.put.ncdf(fid, 0, "Units", "m/s")
   else if(variable == "T")
      att.put.ncdf(fid, 0, "Units", "^oC")

   for(c in 1:length(thresholds)) {
      varname = getPVarName(thresholds[c])
      put.var.ncdf(fid, varname, p[,,,c])
   }
   for(c in 1:length(quantiles)) {
      varname = getQVarName(quantiles[c])
      put.var.ncdf(fid, varname, q[,,,c])
   }
   close.ncdf(fid)
}

getPVarName <- function(thresholds) {
   if(thresholds < 0) {
      varname = paste("pm", -thresholds, sep="")
   }
   else if(thresholds < 1 & thresholds > 0) {
      varname = paste("p0", thresholds*10, sep="")
   }
   else {
      varname = paste("p", thresholds, sep="")
   }
   return(varname)
}

getQVarName <- function(quantiles) {
   varname = paste("q", floor(quantiles*100), sep="")
   return(varname)
}
scatter <- function(fit, x) {
   fcst = pG(0*x$OBS+0.5, fit, x)
   plot(x$P0, fcst)
}

pG <- function(p, fit, x, par=NULL) {
   if(length(p) == 1)
      p = 0*x$OBS + p
   return(getValues(p, fit, x, "p", par))
}
qG <- function(q, fit, x, par=NULL) {
   if(length(q) == 1)
      q = 0*x$OBS + q
   values = getValues(q, fit, x, "q", par)
   return(values)
}
dG <- function(d, fit, x, par=NULL) {
   if(length(d) == 1)
      d = 0*x$OBS + d
   return(getValues(d, fit, x, "d", par))
}
mG <- function(fit, x, par=NULL) {
   return(getValues(0, fit, x, "m", par))
}
getMoments <- function(fit, x) {
   cls = class(fit)
   if(length(cls) != 4 || cls[1] != "gamlss") {
      kens = grep("ENS", names(x))
      par  = t(apply(x[,kens], 1, sort))
      return(par)
   }
   family = fit$family[1]
   par = predictAll(fit, data=x, newdata=x)
   return(par)
}
getValues <- function(q, fit, x, type, par=NULL) {
   if(length(q) != dim(x)[1]) {
      #stop("getValues: q must be the same size as x")
   }
   if(is.character(fit)) {
      if(fit == "raw") {
         values = matrix(NA, nrow=dim(x)[1], ncol=1)
         kens = grep("ENS", names(x))
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
            sites = unique(x$SITE)
            for(site in sites) {
               I = which(x$SITE == site)
               values[I,1] = mean(x$OBS[I] <= q)
            }
         }
         else if(type == "q") {
            sites = unique(x$SITE)
            for(site in sites) {
               I = which(x$SITE == site)
               if(length(I) > 0) {
                  temp  = sort(x$OBS[I])
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
            sites = unique(x$SITE)
            for(site in sites) {
               I = which(x$SITE == site)
               values[I,1] = mean(x$OBS[I])
            }
         }
      }
      else {
         stop()
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
      # qZAGA is really slow, use optimized function
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
