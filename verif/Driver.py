import sys
import os
import verif.Data as Data
import verif.Output as Output
import verif.Metric as Metric
import verif.Common as Common
import verif.Input as Input
import matplotlib.pyplot as mpl
import textwrap
import verif.Version as Version
import numpy as np
def run(argv):
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
   dates = None
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
   version = None

   # Read command line arguments
   i = 1
   while(i < len(argv)):
      arg = argv[i]
      if(arg[0] == '-'):
         # Process option
         if(arg == "-nomargin"):
            noMargin = True
         elif(arg == "--version"):
            version = True
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
               ofile = argv[i+1]
            elif(arg == "-l"):
               locations = Common.parseNumbers(argv[i+1])
            elif(arg == "-llrange"):
               latlonRange = Common.parseNumbers(argv[i+1])
            elif(arg == "-t"):
               training = int(argv[i+1])
            elif(arg == "-x"):
               xdim = argv[i+1]
            elif(arg == "-o"):
               offsets = Common.parseNumbers(argv[i+1])
            elif(arg == "-leg"):
               leg = unicode(argv[i+1], 'utf8')
            elif(arg == "-ylabel"):
               ylabel = unicode(argv[i+1], 'utf8')
            elif(arg == "-xlabel"):
               xlabel = unicode(argv[i+1], 'utf8')
            elif(arg == "-title"):
               title = unicode(argv[i+1], 'utf8')
            elif(arg == "-b"):
               binType = argv[i+1]
            elif(arg == "-type"):
               type = argv[i+1]
            elif(arg == "-fs"):
               figSize = argv[i+1]
            elif(arg == "-dpi"):
               dpi = int(argv[i+1])
            elif(arg == "-d"):
               # Either format is ok:
               # -d 20150101 20150103
               # -d 20150101:20150103
               if(i+2 < len(argv) and argv[i+2].isdigit()):
                  dates = Common.parseNumbers("%s:%s" %(argv[i+1],argv[i+2]), True)
                  i = i + 1
               else:
                  dates = Common.parseNumbers(argv[i+1], True)
            elif(arg == "-c"):
               climFile = argv[i+1]
               climType = "subtract"
            elif(arg == "-C"):
               climFile = argv[i+1]
               climType = "divide"
            elif(arg == "-xlim"):
               xlim = Common.parseNumbers(argv[i+1])
            elif(arg == "-ylim"):
               ylim = Common.parseNumbers(argv[i+1])
            elif(arg == "-clim"):
               clim = Common.parseNumbers(argv[i+1])
            elif(arg == "-s"):
               sdim = argv[i+1]
            elif(arg == "-ct"):
               cType = argv[i+1]
            elif(arg == "-r"):
               thresholds = Common.parseNumbers(argv[i+1])
            elif(arg == "-ms"):
               markerSize = float(argv[i+1])
            elif(arg == "-lw"):
               lineWidth = float(argv[i+1])
            elif(arg == "-tickfs"):
               tickFontSize = float(argv[i+1])
            elif(arg == "-labfs"):
               labFontSize = float(argv[i+1])
            elif(arg == "-legfs"):
               legFontSize = float(argv[i+1])
            elif(arg == "-xrot"):
               XRotation = float(argv[i+1])
            elif(arg == "-majlth"):
               MajorLength = float(argv[i+1])
            elif(arg == "-minlth"):
               MinorLength = float(argv[i+1])
            elif(arg == "-majwid"):
               MajorWidth = float(argv[i+1])
            elif(arg == "-bot"):
               Bottom = float(argv[i+1])
            elif(arg == "-top"):
               Top = float(argv[i+1])
            elif(arg == "-right"):
               Right = float(argv[i+1])
            elif(arg == "-left"):
               Left = float(argv[i+1])
            elif(arg == "-pad"):
               Pad = argv[i+1]
            elif(arg == "-m"):
               metric = argv[i+1]
            else:
               Common.error("Flag '" + argv[i] + "' not recognized")
            i = i + 1
      else:
         ifiles.append(argv[i])
      i = i + 1

   if(version):
      print "Version: " + Version.__version__
      return

   # Deal with legend entries
   if(leg != None):
      leg = leg.split(',')
      for i in range(0,len(leg)):
         leg[i] = leg[i].replace('_', ' ')

   if(latlonRange != None and len(latlonRange) != 4):
      Common.error("-llRange <values> must have exactly 4 values")

   if(len(ifiles) > 0):
      data = Data.Data(ifiles, clim=climFile, climType=climType, dates=dates, offsets=offsets,
            locations=locations, latlonRange=latlonRange, training=training)
   else:
      data = None
   if(len(argv) == 1 or len(ifiles) == 0 or metric == None):
      showDescription(data)
      return

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
      elif(metric == "obs"):
         m = Metric.Obs()
      elif(metric == "fcst"):
         m = Metric.Fcst()
      elif(metric == "rmsf"):
         m = Metric.Rmsf()
      elif(metric == "crmse"):
         m = Metric.Crmse()
      elif(metric == "cmae"):
         m = Metric.Cmae()
      elif(metric == "dmb"):
         m = Metric.Dmb()
      elif(metric == "num"):
         m = Metric.Num()
      elif(metric == "corr"):
         m = Metric.Corr()
      elif(metric == "rankcorr"):
         m = Metric.RankCorr()
      elif(metric == "kendallcorr"):
         m = Metric.KendallCorr()
      elif(metric == "bias"):
         m = Metric.Bias()
      elif(metric == "ef"):
         m = Metric.Ef()
      elif(metric == "stderror"):
         m = Metric.StdError()
      elif(metric == "mae"):
         m = Metric.Mae()
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
         m = Metric.Mean(Metric.Default(metric))

      m.setAggregator(cType)

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
   desc = "Program to compute verification scores for weather forecasts. Can be used to compare forecasts from different files. In that case only dates, offsets, and locations that are common to all forecast files are used."
   print textwrap.fill(desc, Common.getTextWidth())
   print ""
   print "usage: verif files -m metric [options]"
   print "       verif --version"
   print ""
   print Common.green("Arguments:")
   print Common.formatArgument("files", "One or more verification files in NetCDF or text format (see 'File Formats' below).")
   print Common.formatArgument("-m metric","Which verification metric to use? See 'Metrics' below.")
   print Common.formatArgument("--version","What version of verif is this?")
   print ""
   print Common.green("Options:")
   print "Note: vectors can be entered using commas, or MATLAB syntax (i.e 3:5 is 3,4,5 and 3:2:7 is 3,5,7)"
   #print Common.formatArgument("","For vector options, the following are supported:")
   #print Common.formatArgument("","  start:end       e.g. 3:5 gives 3, 4, 5")
   #print Common.formatArgument("","  start:inc:end   e.g. 3:2:7 gives 3, 5, 7")
   #print Common.formatArgument("","  vector1,vector2 e.g. 3:5,1:2 gives 3, 4, 5, 1, 2")
   # Dimensions
   print Common.green("  Dimensions and subset:")
   print Common.formatArgument("-d dates","A vector of dates in YYYYMMDD format, e.g.  20130101:20130201.")
   print Common.formatArgument("-l locations","Limit the verification to these location IDs.")
   print Common.formatArgument("-llrange range","Limit the verification to locations within minlon,maxlon,minlat,maxlat.")
   print Common.formatArgument("-o offsets","Limit the verification to these offsets (in hours).")
   print Common.formatArgument("-r thresholds","Compute scores for these thresholds (only used by some metrics).")
   print Common.formatArgument("-t period","Allow this many days of training, i.e. remove this many days from the beginning of the verification.")
   print Common.formatArgument("-x dim","Plot this dimension on the x-axis: date, offset, location, locationId, locationElev, locationLat, locationLon, threshold, or none. Not supported by all metrics. If not specified, then a default is used based on the metric. 'none' collapses all dimensions and computes one value.")

   # Data manipulation
   print Common.green("  Data manipulation:")
   print Common.formatArgument("-acc","Plot accumulated values. Only works for non-derived metrics")
   print Common.formatArgument("-b type","One of 'below', 'within', or 'above'. For threshold plots (ets, hit, within, etc) 'below/above' computes frequency below/above the threshold, and 'within' computes the frequency between consecutive thresholds.")
   print Common.formatArgument("-c file","File containing climatology data. Subtract all forecasts and obs with climatology values.")
   print Common.formatArgument("-C file","File containing climatology data. Divide all forecasts and obs by climatology values.")
   print Common.formatArgument("-ct type","Collapsing type: 'min', 'mean', 'median', 'max', 'std', and 'range'. Some metrics computes a value for each value on the x-axis. Which function should be used to do the collapsing? Default is 'mean'. Only supported by some metrics.")
   print Common.formatArgument("-hist","Plot values as histogram. Only works for non-derived metrics")
   print Common.formatArgument("-sort","Plot values sorted. Only works for non-derived metrics")

   # Plot options
   print Common.green("  Plotting options:")
   print Common.formatArgument("-bot value","Bottom boundary location for saved figure [range 0-1]")
   print Common.formatArgument("-clim limits","Force colorbar limits to the two values lower,upper")
   print Common.formatArgument("-dpi value","Resolution of image in dots per inch (default 100)")
   print Common.formatArgument("-f file","Save image to this filename")
   print Common.formatArgument("-fs size","Set figure size width,height (in inches). Default 8x6.")
   print Common.formatArgument("-leg titles","Comma-separated list of legend titles. Use '_' to represent space.")
   print Common.formatArgument("-lw width","How wide should lines be?")
   print Common.formatArgument("-labfs size","Font size for axis labels")
   print Common.formatArgument("-left value","Left boundary location for saved figure [range 0-1]")
   print Common.formatArgument("-legfs size","Font size for legend")
   print Common.formatArgument("-majlth length","Length of major tick marks")
   print Common.formatArgument("-majtwid width","Adjust the thickness of the major tick marks")
   print Common.formatArgument("-minlth length","Length of minor tick marks")
   print Common.formatArgument("-ms size","How big should markers be?")
   print Common.formatArgument("-nomargin","Remove margins (whitespace) in the plot not x[i] <= T.")
   print Common.formatArgument("-right value","Right boundary location for saved figure [range 0-1]")
   print Common.formatArgument("-sp","Show a line indicating the perfect score")
   print Common.formatArgument("-tickfs size","Font size for axis ticks")
   print Common.formatArgument("-title text","Custom title to chart top")
   print Common.formatArgument("-top value","Top boundary location for saved figure [range 0-1]")
   print Common.formatArgument("-type type","One of 'plot' (default), 'text', 'map', or 'maprank'.")
   print Common.formatArgument("-xlabel text","Custom x-axis label")
   print Common.formatArgument("-xlim limits","Force x-axis limits to the two values lower,upper")
   print Common.formatArgument("-xrot value","Rotation angle for x-axis labels")
   print Common.formatArgument("-ylabel text","Custom y-axis label")
   print Common.formatArgument("-ylim limits","Force y-axis limits to the two values lower,upper")
   print ""
   metrics = Metric.getAllMetrics()
   outputs = Output.getAllOutputs()
   print Common.green("Metrics (-m):")
   metricOutputs = metrics + outputs
   metricOutputs.sort(key=lambda x: x[0].lower(), reverse=False)
   for m in metricOutputs:
      name = m[0].lower()
      desc = m[1].summary()
      if(desc != ""):
         print Common.formatArgument(name, desc)
         #print "   %-14s%s" % (name, textwrap.fill(desc, 80).replace('\n', '\n                 ')),
         #print ""
   if(data != None):
      print ""
      print "  Or one of the following, which plots the raw score from the file:"
      print " ",
      metrics = data.getMetrics()
      for metric in metrics:
         print metric,
   print ""
   print ""
   print Common.green("File formats:")
   print Input.Text.description()
   print Input.Comps.description()

if __name__ == '__main__':
       main()
