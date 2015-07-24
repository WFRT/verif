import sys
import os
import verif.Data as Data
import verif.Output as Output
import verif.Metric as Metric
import verif.Common as Common
import matplotlib.pyplot as mpl
import textwrap
def main():
   ############
   # Defaults #
   ############
   ifiles   = list()
   ofile    = None
   metric   = None
   locations = None
   latlonRange = None
   training = 0
   thresholds = None
   startDate = None
   endDate   = None
   climFile   = None
   climType = "subtract"
   leg    = None
   ylabel = None
   xlabel = None
   title  = None
   offsets = None
   xdim = None
   sdim = None
   figSize = None
   dpi     = 100
   debug = False
   showText = False
   showMap = False
   noMargin = False
   binType     = None
   markerSize = None
   lineWidth = None
   tickFontSize  = None
   labFontSize  = None
   legFontSize  = None
   type = "plot"
   XRotation = None
   MajorLength = None
   MinorLength = None
   MajorWidth = None
   Bottom = None
   Top = None
   Right = None
   Left = None
   Pad = None
   showPerfect = None
   cType = "mean"
   doHist = False
   doSort = False
   doAcc  = False
   xlim = None
   ylim = None
   clim = None

   # Read command line arguments
   i = 1
   while(i < len(sys.argv)):
      arg = sys.argv[i]
      if(arg[0] == '-'):
         # Process option
         if(arg == "-debug"):
            debug = True
         elif(arg == "-nomargin"):
            noMargin = True
         elif(arg == "-sp"):
            showPerfect = True
         elif(arg == "-hist"):
            doHist = True
         elif(arg == "-acc"):
            doAcc = True
         elif(arg == "-sort"):
            doSort = True
         else:
            if(arg == "-f"):
               ofile = sys.argv[i+1]
            elif(arg == "-l"):
               locations = Common.parseNumbers(sys.argv[i+1])
            elif(arg == "-llrange"):
               latlonRange = Common.parseNumbers(sys.argv[i+1])
            elif(arg == "-t"):
               training = int(sys.argv[i+1])
            elif(arg == "-x"):
               xdim = sys.argv[i+1]
            elif(arg == "-o"):
               offsets = Common.parseNumbers(sys.argv[i+1])
            elif(arg == "-leg"):
               leg = unicode(sys.argv[i+1], 'utf8')
            elif(arg == "-ylabel"):
               ylabel = unicode(sys.argv[i+1], 'utf8')
            elif(arg == "-xlabel"):
               xlabel = unicode(sys.argv[i+1], 'utf8')
            elif(arg == "-title"):
               title = unicode(sys.argv[i+1], 'utf8')
            elif(arg == "-b"):
               binType = sys.argv[i+1]
            elif(arg == "-type"):
               type = sys.argv[i+1]
            elif(arg == "-fs"):
               figSize = sys.argv[i+1]
            elif(arg == "-dpi"):
               dpi = int(sys.argv[i+1])
            elif(arg == "-d"):
               startDate = int(sys.argv[i+1])
               endDate   = int(sys.argv[i+2])
               i = i + 1
            elif(arg == "-c"):
               climFile = sys.argv[i+1]
               climType = "subtract"
            elif(arg == "-C"):
               climFile = sys.argv[i+1]
               climType = "divide"
            elif(arg == "-xlim"):
               xlim = Common.parseNumbers(sys.argv[i+1])
            elif(arg == "-ylim"):
               ylim = Common.parseNumbers(sys.argv[i+1])
            elif(arg == "-clim"):
               clim = Common.parseNumbers(sys.argv[i+1])
            elif(arg == "-s"):
               sdim = sys.argv[i+1]
            elif(arg == "-ct"):
               cType = sys.argv[i+1]
            elif(arg == "-r"):
               thresholds = Common.parseNumbers(sys.argv[i+1])
            elif(arg == "-ms"):
               markerSize = float(sys.argv[i+1])
            elif(arg == "-lw"):
               lineWidth = float(sys.argv[i+1])
            elif(arg == "-tickfs"):
               tickFontSize = float(sys.argv[i+1])
            elif(arg == "-labfs"):
               labFontSize = float(sys.argv[i+1])
            elif(arg == "-legfs"):
               legFontSize = float(sys.argv[i+1])
            elif(arg == "-xrot"):
               XRotation = float(sys.argv[i+1])
            elif(arg == "-majlth"):
               MajorLength = float(sys.argv[i+1])
            elif(arg == "-minlth"):
               MinorLength = float(sys.argv[i+1])
            elif(arg == "-majwid"):
               MajorWidth = float(sys.argv[i+1])
            elif(arg == "-bot"):
               Bottom = float(sys.argv[i+1])
            elif(arg == "-top"):
               Top = float(sys.argv[i+1])
            elif(arg == "-right"):
               Right = float(sys.argv[i+1])
            elif(arg == "-left"):
               Left = float(sys.argv[i+1])
            elif(arg == "-pad"):
               Pad = sys.argv[i+1]
            elif(arg == "-m"):
               metric = sys.argv[i+1]
            else:
               Common.error("Flag '" + sys.argv[i] + "' not recognized")
            i = i + 1
      else:
         ifiles.append(sys.argv[i])
      i = i + 1

   # Deal with legend entries
   if(leg != None):
      leg = leg.split(',')
      for i in range(0,len(leg)):
         leg[i] = leg[i].replace('_', ' ')

   # Limit dates
   dates = None
   if(startDate != None and endDate != None):
      dates = list()
      date = startDate
      while(date <= endDate):
         dates.append(date)
         date = Common.getDate(date, 1)

   if(cType != "mean" and cType != "min" and cType != "max" and cType != "median"):
      Common.error("'-ct cType' must be one of min, mean, median, or max")

   if(latlonRange != None and len(latlonRange) != 4):
      Common.error("-llRange <values> must have exactly 4 values")

   if(len(ifiles) > 0):
      data = Data.Data(ifiles, clim=climFile, climType=climType, dates=dates, offsets=offsets,
            locations=locations, latlonRange=latlonRange, training=training)
   else:
      data = None
   if(len(sys.argv) == 1 or len(ifiles) == 0 or metric == None):
      showDescription(data)
      sys.exit()

   if(figSize != None):
      figSize = figSize.split(',')
      if(len(figSize) != 2):
         print "-fs figSize must be in the form: width,height"
         sys.exit(1)

   m = None

   # Handle special plots
   if(doHist):
      pl = Output.Hist(metric)
   elif(doSort):
      pl = Output.Sort(metric)
   elif(metric == "pithist"):
      m  = Metric.Pit("pit")
      pl = Output.PitHist(m)
   elif(metric == "obsfcst"):
      pl = Output.ObsFcst()
   elif(metric == "timeseries"):
      pl = Output.TimeSeries()
   elif(metric == "qq"):
      pl = Output.QQ()
   elif(metric == "cond"):
      pl = Output.Cond()
   elif(metric == "against"):
      pl = Output.Against()
   elif(metric == "count"):
      pl = Output.Count()
   elif(metric == "scatter"):
      pl = Output.Scatter()
   elif(metric == "change"):
      pl = Output.Change()
   elif(metric == "spreadskill"):
      pl = Output.SpreadSkill()
   elif(metric == "taylor"):
      pl = Output.Taylor()
   elif(metric == "error"):
      pl = Output.Error()
   elif(metric == "droc"):
      pl = Output.DRoc()
   elif(metric == "droc0"):
      pl = Output.DRoc0()
   elif(metric == "drocnorm"):
      pl = Output.DRocNorm()
   elif(metric == "reliability"):
      pl = Output.Reliability()
   elif(metric == "invreliability"):
      pl = Output.InvReliability()
   elif(metric == "igncontrib"):
      pl = Output.IgnContrib()
   elif(metric == "marginal"):
      pl = Output.Marginal()
   else:
      # Standard plots
      '''
      # Attempt at automating
      metrics = Metric.getAllMetrics()
      m = None
      for mm in metrics:
         if(metric == mm[0].lower() and mm[1].isStandard()):
            m = mm[1]()
            break
      if(m == None):
         m = Metric.Default(metric)
         '''

      # Determine metric
      if(metric == "rmse"):
         m = Metric.Rmse()
      elif(metric == "rmsf"):
         m = Metric.Rmsf()
      elif(metric == "crmse"):
         m = Metric.Crmse()
      elif(metric == "cmae"):
         m = Metric.Cmae()
      elif(metric == "dmb"):
         m = Metric.Dmb()
      elif(metric == "std"):
         m = Metric.Std()
      elif(metric == "num"):
         m = Metric.Num()
      elif(metric == "corr"):
         m = Metric.Corr()
      elif(metric == "rankcorr"):
         m = Metric.RankCorr()
      elif(metric == "bias"):
         m = Metric.Bias()
      elif(metric == "ef"):
         m = Metric.Ef()
      elif(metric == "maxobs"):
         m = Metric.MaxObs()
      elif(metric == "minobs"):
         m = Metric.MinObs()
      elif(metric == "maxfcst"):
         m = Metric.MaxFcst()
      elif(metric == "minfcst"):
         m = Metric.MinFcst()
      elif(metric == "stderror"):
         m = Metric.StdError()
      elif(metric == "mae"):
         m = Metric.Mae()
      elif(metric == "medae"):
         m = Metric.Medae()
      # Contingency metrics
      elif(metric == "ets"):
         m = Metric.Ets()
      elif(metric == "threat"):
         m = Metric.Threat()
      elif(metric == "pc"):
         m = Metric.Pc()
      elif(metric == "diff"):
         m = Metric.Diff()
      elif(metric == "edi"):
         m = Metric.Edi()
      elif(metric == "sedi"):
         m = Metric.Sedi()
      elif(metric == "eds"):
         m = Metric.Eds()
      elif(metric == "seds"):
         m = Metric.Seds()
      elif(metric == "biasfreq"):
         m = Metric.BiasFreq()
      elif(metric == "hss"):
         m = Metric.Hss()
      elif(metric == "baserate"):
         m = Metric.BaseRate()
      elif(metric == "yulesq"):
         m = Metric.YulesQ()
      elif(metric == "or"):
         m = Metric.Or()
      elif(metric == "lor"):
         m = Metric.Lor()
      elif(metric == "yulesq"):
         m = Metric.YulesQ()
      elif(metric == "kss"):
         m = Metric.Kss()
      elif(metric == "hit"):
         m = Metric.Hit()
      elif(metric == "miss"):
         m = Metric.Miss()
      elif(metric == "fa"):
         m = Metric.Fa()
      elif(metric == "far"):
         m = Metric.Far()
      # Other threshold
      elif(metric == "bs"):
         m = Metric.Bs()
      elif(metric == "bss"):
         m = Metric.Bss()
      elif(metric == "bsrel"):
         m = Metric.BsRel()
      elif(metric == "bsunc"):
         m = Metric.BsUnc()
      elif(metric == "bsres"):
         m = Metric.BsRes()
      elif(metric == "ign0"):
         m = Metric.Ign0()
      elif(metric == "spherical"):
         m = Metric.Spherical()
      elif(metric == "within"):
         m = Metric.Within()
      # Probabilistic
      elif(metric == "pit"):
         m = Metric.Mean(Metric.Pit())
      elif(metric == "pitdev"):
         m = Metric.PitDev()
      elif(metric == "marginalratio"):
         m = Metric.MarginalRatio()
      # Default
      else:
         if(cType == "min"):
            m = Metric.Min(Metric.Default(metric))
         elif(cType == "max"):
            m = Metric.Max(Metric.Default(metric))
         elif(cType == "median"):
            m = Metric.Median(Metric.Default(metric))
         elif(cType == "mean"):
            m = Metric.Mean(Metric.Default(metric))
         else:
            Common.error("-ct " + cType + " not understood")

      # Output type
      if(type == "plot" or type == "text" or type == "map" or type == "maprank"):
         pl = Output.Default(m)
         pl.setShowAcc(doAcc)
      else:
         Common.error("Type not understood")

   # Rest dimension of '-x' is not allowed
   if(xdim != None and not pl.supportsX()):
      Common.warning(metric + " does not support -x. Ignoring it.")
      xdim = None

   # Reset dimension if 'threshold' is not allowed
   if(xdim == "threshold" and ((not pl.supportsThreshold()) or (not m.supportsThreshold()))):
      Common.warning(metric + " does not support '-x threshold'. Ignoring it.")
      thresholds = None
      xdim = None

   # Create thresholds if needed
   if((thresholds == None) and (pl.requiresThresholds() or (m != None and m.requiresThresholds()))):
      data.setAxis("none")
      obs  = data.getScores("obs")[0]
      fcst = data.getScores("fcst")[0]
      smin = min(min(obs), min(fcst))
      smax = max(max(obs), max(fcst))
      thresholds = np.linspace(smin,smax,10)
      Common.warning("Missing '-r <thresholds>'. Automatically setting thresholds.")

   # Set plot parameters
   if(markerSize != None):
      pl.setMarkerSize(markerSize)
   if(lineWidth != None):
      pl.setLineWidth(lineWidth)
   if(labFontSize != None):
      pl.setLabFontSize(labFontSize)
   if(legFontSize != None):
      pl.setLegFontSize(legFontSize)
   if(tickFontSize != None):
      pl.setTickFontSize(tickFontSize)
   if(XRotation != None):
      pl.setXRotation(XRotation)
   if(MajorLength != None):
      pl.setMajorLength(MajorLength)
   if(MinorLength != None):
      pl.setMinorLength(MinorLength)
   if(MajorWidth != None):
      pl.setMajorWidth(MajorWidth)
   if(Bottom != None):
      pl.setBottom(Bottom)
   if(Top != None):
      pl.setTop(Top)
   if(Right != None):
      pl.setRight(Right)
   if(Left != None):
      pl.setLeft(Left)
   if(Pad != None):
      pl.setPad(None)
   if(binType != None):
      pl.setBinType(binType)
   if(showPerfect != None):
      pl.setShowPerfect(showPerfect)
   if(xlim != None):
      pl.setXLim(xlim)
   if(ylim != None):
      pl.setYLim(ylim)
   if(clim != None):
      pl.setCLim(clim)
   pl.setFilename(ofile)
   pl.setThresholds(thresholds)
   pl.setLegend(leg)
   pl.setFigsize(figSize)
   pl.setDpi(dpi)
   pl.setAxis(xdim)
   pl.setShowMargin(not noMargin)
   pl.setYlabel(ylabel)
   pl.setXlabel(xlabel)
   pl.setTitle(title)

   if(type == "text"):
      pl.text(data)
   elif(type == "map"):
      pl.map(data)
   elif(type == "maprank"):
      pl.setShowRank(True)
      pl.map(data)
   else:
      pl.plot(data)

def showDescription(data=None):
   print "Compute verification scores for COMPS verification files\n"
   print "usage: verif files -m metric [-x x-dim] [-r thresholds]"
   print "                 [-l locationIds] [-llrange latLonRange]"
   print "                 [-o offsets] [-d start-date end-date]"
   print "                 [-t training] [-c climFile] [-C ClimFile]"
   print "                 [-xlim xlim] [-ylim ylim] [-c clim]"
   print "                 [-type type] [-leg legend] [-hist] [-sort] [-acc]"
   print "                 [-f imageFile] [-fs figSize] [-dpi dpi]"
   print "                 [-b binType] [-nomargin] [-debug] [-ct ctype]"
   print "                 [-ms markerSize] [-lw lineWidth] [-xrot XRotation]"
   print "                 [-tickfs tickFontSize] [-labfs labFontSize] [-legfs legFontSize]"
   print "                 [-majlth MajorTickLength] [-minlth MinorTickLength] [-majwid MajorTickWidth]"
   print "                 [-bot Bottom] [-top Top] [-left Left] [-right Right]"
   print "                 [-sp] [-ylabel Y Axis Label] [-xlabel X Axis Label]"
   print "		   [-title Title]"
   #print "                 [-pad Pad]"
   print ""
   print Common.green("Arguments:")
   print "   files         One or more COMPS verification files in NetCDF format."
   print "   metric        Verification score to use. See available metrics below."
   print "   x-dim         Plot this dimension on the x-axis: date, offset, location, locationId,"
   print "                 locationElev, locationLat, locationLon, threshold, or none. Not supported by"
   print "                 all metrics. If not specified, then a default is used based on the metric."
   print "                 'none' collapses all dimensions and computes one value."
   print "   thresholds    Compute scores for these thresholds (only used by some metrics)."
   print "   locationIds   Limit the verification to these location IDs."
   print "   latLonRange   Limit the verification to locations within minlon,maxlon,minlat,maxlat."
   print "   offsets       Limit the verification to these offsets (in hours)."
   print "   start-date    YYYYMMDD. Only use dates from this day and on"
   print "   end-date      YYYYMMDD. Only use dates up to and including this day"
   print "   training      Remove this many days from the beginning of the verification."
   print "   climFile      NetCDF containing climatology data. Subtract all forecasts and"
   print "                 obs with climatology values."
   print "   ClimFile      NetCDF containing climatology data. Divide all forecasts and"
   print "                 obs by climatology values."
   print "   xlim          Force x-axis limits to the two values lower,upper"
   print "   ylim          Force y-axis limits to the two values lower,upper"
   print "   clim          Force colorbar limits to the two values lower,upper"
   print "   type          One of 'plot' (default),'text', 'map', or 'maprank'."
   print "   -hist         Plot values as histogram. Only works for non-derived metrics"
   print "   -sort         Plot values sorted. Only works for non-derived metrics"
   print "   -acc          Plot accumulated values. Only works for non-derived metrics"
   print "   legend        Comma-separated list of legend titles. Use '_' to represent space."
   print "   imageFile     Save image to this filename"
   print "   figSize       Set figure size width,height (in inches). Default 8x6."
   print "   dpi           Resolution of image in dots per inch (default 100)"
   print "   binType       One of 'below', 'within', or 'above'. For threshold plots (ets, hit, within, etc)"
   print "                 'below/above' computes frequency below/above the threshold, and 'within' computes"
   print "                 the frequency between consecutive thresholds."
   print "   -nomargin     Remove margins (whitespace) in the plot"
   print "                 not x[i] <= T."
   print "   -debug        Show statistics about files"
   print "   cType         Collapsing type: 'min', 'mean', 'median', or 'max. When a score from the file is plotted"
   print "                 (such as -m 'fcst'), the min/mean/meadian/max will be shown for each value on the x-axis"
   print "   markerSize    How big should markers be?"
   print "   lineWidth     How wide should lines be?"
   print "   XRotation     Rotation angle for x-axis labels"
   print "   tickFontSize  Font size for axis ticks"
   print "   labFontSize   Font size for axis labels"
   print "   legFontSize   Font size for legend"
   print "   MajorTickLength  Length of major tick marks"
   print "   MinorTickLength  Length of minor tick marks"
   print "   MajorTickWidth   Adjust the thickness of the major tick marks"
   print "   Bottom        Bottom boundary location for saved figure [range 0-1]"
   print "   Top           Top boundary location for saved figure [range 0-1]"
   print "   Left          Left boundary location for saved figure [range 0-1]"
   print "   Right         Right boundary location for saved figure [range 0-1]"
   print "   -sp           Show a line indicating the perfect score"
   print "   -ylabel	   Custom Y-axis Label"
   print "   -xlabel       Custom X-axis Label"
   print "   -title        Custom Title to Chart Top"
   print ""
   metrics = Metric.getAllMetrics()
   outputs = Output.getAllOutputs()
   print Common.green("Metrics (-m):")
   for m in metrics+outputs:
      name = m[0].lower()
      desc = m[1].summary()
      if(desc != ""):
         print "   %-14s%s" % (name, textwrap.fill(desc, 80).replace('\n', '\n                 ')),
         print ""
   if(data != None):
      print ""
      print "   Or one of the following, which plots the raw score from the file:"
      metrics = data.getMetrics()
      for metric in metrics:
         print "   " + metric

if __name__ == '__main__':
       main()
