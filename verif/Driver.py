import sys
import os
import verif.Data as Data
import verif.Output as Output
import verif.Metric as Metric
import verif.Util as Util
import verif.Input as Input
import matplotlib.pyplot as mpl
import textwrap
import verif.Version as Version
import numpy as np


def run(argv):
   ############
   # Defaults #
   ############
   ifiles = list()
   ofile = None
   metric = None
   locations = None
   latlonRange = None
   elevRange = None
   training = 0
   thresholds = None
   dates = None
   climFile = None
   climType = "subtract"
   leg = None
   ylabel = None
   xlabel = None
   title = None
   offsets = None
   xdim = None
   sdim = None
   figSize = None
   dpi = 100
   showText = False
   showMap = False
   noMargin = False
   binType = None
   simple = None
   markerSize = None
   lineWidth = None
   lineColors = None
   tickFontSize = None
   labFontSize = None
   legFontSize = None
   legLoc = None
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
   aggregatorName = "mean"
   doHist = False
   doSort = False
   doAcc = False
   xlim = None
   ylim = None
   clim = None
   version = None
   listThresholds = False
   listQuantiles = False
   listLocations = False
   listDates = False
   showSatellite = False
   logX = False
   logY = False
   cmap = None

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
         elif(arg == "--list-thresholds"):
            listThresholds = True
         elif(arg == "--list-quantiles"):
            listQuantiles = True
         elif(arg == "--list-locations"):
            listLocations = True
         elif(arg == "--list-dates"):
            listDates = True
         elif(arg == "-sp"):
            showPerfect = True
         elif(arg == "-hist"):
            doHist = True
         elif(arg == "-acc"):
            doAcc = True
         elif(arg == "-sort"):
            doSort = True
         elif(arg == "-simple"):
            simple = True
         elif(arg == "-logx"):
            logX = True
         elif(arg == "-logy"):
            logY = True
         elif(arg == "-sat"):
            showSatellite = True
         else:
            if(arg == "-f"):
               ofile = argv[i + 1]
            elif(arg == "-l"):
               locations = Util.parseNumbers(argv[i + 1])
            elif(arg == "-llrange"):
               latlonRange = Util.parseNumbers(argv[i + 1])
            elif(arg == "-elevrange"):
               elevRange = Util.parseNumbers(argv[i + 1])
            elif(arg == "-t"):
               training = int(argv[i + 1])
            elif(arg == "-x"):
               xdim = argv[i + 1]
            elif(arg == "-o"):
               offsets = Util.parseNumbers(argv[i + 1])
            elif(arg == "-leg"):
               leg = unicode(argv[i + 1], 'utf8')
            elif(arg == "-ylabel"):
               ylabel = unicode(argv[i + 1], 'utf8')
            elif(arg == "-xlabel"):
               xlabel = unicode(argv[i + 1], 'utf8')
            elif(arg == "-title"):
               title = unicode(argv[i + 1], 'utf8')
            elif(arg == "-b"):
               binType = argv[i + 1]
            elif(arg == "-type"):
               type = argv[i + 1]
            elif(arg == "-fs"):
               figSize = argv[i + 1]
            elif(arg == "-dpi"):
               dpi = int(argv[i + 1])
            elif(arg == "-d"):
               # Either format is ok:
               # -d 20150101 20150103
               # -d 20150101:20150103
               if(i + 2 < len(argv) and argv[i + 2].isdigit()):
                  dates = Util.parseNumbers("%s:%s" % (argv[i + 1],
                     argv[i + 2]), True)
                  i = i + 1
               else:
                  dates = Util.parseNumbers(argv[i + 1], True)
            elif(arg == "-c"):
               climFile = argv[i + 1]
               climType = "subtract"
            elif(arg == "-C"):
               climFile = argv[i + 1]
               climType = "divide"
            elif(arg == "-xlim"):
               xlim = Util.parseNumbers(argv[i + 1])
            elif(arg == "-ylim"):
               ylim = Util.parseNumbers(argv[i + 1])
            elif(arg == "-clim"):
               clim = Util.parseNumbers(argv[i + 1])
            elif(arg == "-s"):
               sdim = argv[i + 1]
            elif(arg == "-ct"):
               aggregatorName = argv[i + 1]
            elif(arg == "-r"):
               thresholds = Util.parseNumbers(argv[i + 1])
            elif(arg == "-ms"):
               markerSize = float(argv[i + 1])
            elif(arg == "-lw"):
               lineWidth = float(argv[i + 1])
            elif(arg == "-lc"):
               lineColors = argv[i + 1]
            elif(arg == "-tickfs"):
               tickFontSize = float(argv[i + 1])
            elif(arg == "-labfs"):
               labFontSize = float(argv[i + 1])
            elif(arg == "-legfs"):
               legFontSize = float(argv[i + 1])
            elif(arg == "-legloc"):
               legLoc = argv[i + 1].replace('_', ' ')
            elif(arg == "-xrot"):
               XRotation = float(argv[i + 1])
            elif(arg == "-majlth"):
               MajorLength = float(argv[i + 1])
            elif(arg == "-minlth"):
               MinorLength = float(argv[i + 1])
            elif(arg == "-majwid"):
               MajorWidth = float(argv[i + 1])
            elif(arg == "-bot"):
               Bottom = float(argv[i + 1])
            elif(arg == "-top"):
               Top = float(argv[i + 1])
            elif(arg == "-right"):
               Right = float(argv[i + 1])
            elif(arg == "-left"):
               Left = float(argv[i + 1])
            elif(arg == "-pad"):
               Pad = argv[i + 1]
            elif(arg == "-cmap"):
               cmap = argv[i + 1]
            elif(arg == "-m"):
               metric = argv[i + 1]
            else:
               Util.error("Flag '" + argv[i] + "' not recognized")
            i = i + 1
      else:
         ifiles.append(argv[i])
      i = i + 1

   if(version):
      print "Version: " + Version.__version__
      return

   # Deal with legend entries
   if(leg is not None):
      leg = leg.split(',')
      for i in range(0, len(leg)):
         leg[i] = leg[i].replace('_', ' ')

   if(latlonRange is not None and len(latlonRange) != 4):
      Util.error("-llRange <values> must have exactly 4 values")

   if(elevRange is not None and len(elevRange) != 2):
      Util.error("-elevRange <values> must have exactly 2 values")

   if(len(ifiles) > 0):
      data = Data.Data(ifiles, clim=climFile, climType=climType, dates=dates,
            offsets=offsets, locations=locations, latlonRange=latlonRange,
            elevRange=elevRange, training=training, legend=leg)
   else:
      data = None

   if(listThresholds or listQuantiles or listLocations or listDates):
      if(len(ifiles) == 0):
         Util.error("Files are required in order to list thresholds or quantiles")
      if(listThresholds):
         print "Thresholds:",
         for threshold in data.getThresholds():
            print "%g" % threshold,
         print ""
      if(listQuantiles):
         print "Quantiles:",
         for quantile in data.getQuantiles():
            print "%g" % quantile,
         print ""
      if(listLocations):
         print "    id     lat     lon    elev"
         for station in data.getStations():
            print "%6d %7.2f %7.2f %7.1f" % (station.id(), station.lat(),
                  station.lon(), station.elev())
         print ""
      if(listDates):
         dates = data.getAxisValues("date")
         dates = Util.convertToYYYYMMDD(dates)
         for date in dates:
            print "%d" % date
         print ""
      return
   elif(len(ifiles) == 0 and metric is not None):
      m = Metric.getMetric(metric)
      if(m is not None):
         print m.help()
      else:
         m = Output.getOutput(metric)
         if(m is not None):
            print m.help()
      return
   elif(len(argv) == 1 or len(ifiles) == 0 or metric is None):
      showDescription(data)
      return

   if(figSize is not None):
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
      m = Metric.Pit("pit")
      pl = Output.PitHist(m)
   elif(metric == "obsfcst"):
      pl = Output.ObsFcst()
   elif(metric == "timeseries"):
      pl = Output.TimeSeries()
   elif(metric == "meteo"):
      pl = Output.Meteo()
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
   elif(metric == "freq"):
      pl = Output.Freq()
   elif(metric == "roc"):
      pl = Output.Roc()
   elif(metric == "droc"):
      pl = Output.DRoc()
   elif(metric == "droc0"):
      pl = Output.DRoc0()
   elif(metric == "drocnorm"):
      pl = Output.DRocNorm()
   elif(metric == "reliability"):
      pl = Output.Reliability()
   elif(metric == "discrimination"):
      pl = Output.Discrimination()
   elif(metric == "performance"):
      pl = Output.Performance()
   elif(metric == "invreliability"):
      pl = Output.InvReliability()
   elif(metric == "igncontrib"):
      pl = Output.IgnContrib()
   elif(metric == "economicvalue"):
      pl = Output.EconomicValue()
   elif(metric == "marginal"):
      pl = Output.Marginal()
   else:
      # Standard plots
      # Attempt at automating
      m = Metric.getMetric(metric)
      if(m is None):
         m = Metric.Default(metric)

      m.setAggregator(aggregatorName)

      # Output type
      if(type == "plot" or type == "text" or type == "map" or
            type == "maprank"):
         pl = Output.Default(m)
         pl.setShowAcc(doAcc)
      else:
         Util.error("Type not understood")

   # Rest dimension of '-x' is not allowed
   if(xdim is not None and not pl.supportsX()):
      Util.warning(metric + " does not support -x. Ignoring it.")
      xdim = None

   # Reset dimension if 'threshold' is not allowed
   if(xdim == "threshold" and
         ((not pl.supportsThreshold()) or (m is not None and not m.supportsThreshold()))):
      Util.warning(metric + " does not support '-x threshold'. Ignoring it.")
      thresholds = None
      xdim = None

   # Create thresholds if needed
   if((thresholds is None) and (pl.requiresThresholds() or
         (m is not None and m.requiresThresholds()))):
      data.setAxis("none")
      obs = data.getScores("obs")[0]
      fcst = data.getScores("fcst")[0]
      smin = min(min(obs), min(fcst))
      smax = max(max(obs), max(fcst))
      thresholds = np.linspace(smin, smax, 10)
      Util.warning("Missing '-r <thresholds>'. Automatically setting\
            thresholds.")

   # Set plot parameters
   if(simple is not None):
      pl.setSimple(simple)
   if(markerSize is not None):
      pl.setMarkerSize(markerSize)
   if(lineWidth is not None):
      pl.setLineWidth(lineWidth)
   if(lineColors is not None):
      pl.setLineColors(lineColors)
   if(labFontSize is not None):
      pl.setLabFontSize(labFontSize)
   if(legFontSize is not None):
      pl.setLegFontSize(legFontSize)
   if(legLoc is not None):
      pl.setLegLoc(legLoc)
   if(tickFontSize is not None):
      pl.setTickFontSize(tickFontSize)
   if(XRotation is not None):
      pl.setXRotation(XRotation)
   if(MajorLength is not None):
      pl.setMajorLength(MajorLength)
   if(MinorLength is not None):
      pl.setMinorLength(MinorLength)
   if(MajorWidth is not None):
      pl.setMajorWidth(MajorWidth)
   if(Bottom is not None):
      pl.setBottom(Bottom)
   if(Top is not None):
      pl.setTop(Top)
   if(Right is not None):
      pl.setRight(Right)
   if(Left is not None):
      pl.setLeft(Left)
   if(Pad is not None):
      pl.setPad(None)
   if(binType is not None):
      pl.setBinType(binType)
   if(showPerfect is not None):
      pl.setShowPerfect(showPerfect)
   if(xlim is not None):
      pl.setXLim(xlim)
   if(ylim is not None):
      pl.setYLim(ylim)
   if(clim is not None):
      pl.setCLim(clim)
   if(logX is not None):
      pl.setLogX(logX)
   if(logY is not None):
      pl.setLogY(logY)
   if(cmap is not None):
      pl.setCmap(cmap)
   pl.setFilename(ofile)
   pl.setThresholds(thresholds)
   pl.setFigsize(figSize)
   pl.setDpi(dpi)
   pl.setAxis(xdim)
   pl.setAggregatorName(aggregatorName)
   pl.setShowMargin(not noMargin)
   pl.setYlabel(ylabel)
   pl.setXlabel(xlabel)
   pl.setTitle(title)
   pl.setShowSatellite(showSatellite)

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
   desc = "Program to compute verification scores for weather forecasts. Can be " \
          "used to compare forecasts from different files. In that case only dates, "\
          "offsets, and locations that are common to all forecast files are used."
   print textwrap.fill(desc, Util.getTextWidth())
   print ""
   print "usage: verif files -m metric [options]"
   print "       verif files [--list-thresholds] [--list-quantiles] [--list-locations]"
   print "       verif --version"
   print ""
   print Util.green("Arguments:")
   print Util.formatArgument("files", "One or more verification files in NetCDF or text format (see 'File Formats' below).")
   print Util.formatArgument("-m metric", "Which verification metric to use? See 'Metrics' below.")
   print Util.formatArgument("--list-thresholds", "What thresholds are available in the files?")
   print Util.formatArgument("--list-quantiles", "What quantiles are available in the files?")
   print Util.formatArgument("--list-locations", "What locations are available in the files?")
   print Util.formatArgument("--version", "What version of verif is this?")
   print ""
   print Util.green("Options:")
   print "Note: vectors can be entered using commas, or MATLAB syntax (i.e 3:5 is 3,4,5 and 3:2:7 is 3,5,7)"
   # Dimensions
   print Util.green("  Dimensions and subset:")
   print Util.formatArgument("-elevrange range", "Limit the verification to locations within minelev,maxelev.")
   print Util.formatArgument("-d dates", "A vector of dates in YYYYMMDD format, e.g.  20130101:20130201.")
   print Util.formatArgument("-l locations", "Limit the verification to these location IDs.")
   print Util.formatArgument("-llrange range", "Limit the verification to locations within minlon,maxlon,minlat,maxlat.")
   print Util.formatArgument("-o offsets", "Limit the verification to these offsets (in hours).")
   print Util.formatArgument("-r thresholds", "Compute scores for these thresholds (only used by some metrics).")
   print Util.formatArgument("-t period", "Allow this many days of training, i.e. remove this many days from the beginning of the verification.")
   print Util.formatArgument("-x dim", "Plot this dimension on the x-axis: date, offset, year, month, location, locationId, locationElev, locationLat, locationLon, threshold, or none. Not supported by all metrics. If not specified, then a default is used based on the metric. 'none' collapses all dimensions and computes one value.")

   # Data manipulation
   print Util.green("  Data manipulation:")
   print Util.formatArgument("-acc", "Plot accumulated values. Only works for non-derived metrics")
   print Util.formatArgument("-b type", "One of 'below', 'within', or 'above'. For threshold plots (ets, hit, within, etc) 'below/above' computes frequency below/above the threshold, and 'within' computes the frequency between consecutive thresholds.")
   print Util.formatArgument("-c file", "File containing climatology data. Subtract all forecasts and obs with climatology values.")
   print Util.formatArgument("-C file", "File containing climatology data. Divide all forecasts and obs by climatology values.")
   print Util.formatArgument("-ct type", "Collapsing type: 'count', 'min', 'mean', 'meanabs', 'median', 'max', 'range', 'std', or a number between 0 and 1. Some metrics computes a value for each value on the x-axis. Which function should be used to do the collapsing? Default is 'mean'. 'meanabs' is the mean absolute value. Only supported by some metrics. A number between 0 and 1 returns a specific quantile (e.g.  0.5 is the median)")
   print Util.formatArgument("-hist", "Plot values as histogram. Only works for non-derived metrics")
   print Util.formatArgument("-sort", "Plot values sorted. Only works for non-derived metrics")

   # Plot options
   print Util.green("  Plotting options:")
   print Util.formatArgument("-bot value", "Bottom boundary location for saved figure [range 0-1]")
   print Util.formatArgument("-clim limits", "Force colorbar limits to the two values lower,upper")
   print Util.formatArgument("-cmap colormap", "Use this colormap when possible (e.g. jet, inferno, RdBu)")
   print Util.formatArgument("-dpi value", "Resolution of image in dots per inch (default 100)")
   print Util.formatArgument("-f file", "Save image to this filename")
   print Util.formatArgument("-fs size", "Set figure size width,height (in inches). Default 8x6.")
   print Util.formatArgument("-labfs size", "Font size for axis labels")
   print Util.formatArgument("-lc colors", "Comma-separated list of line colors, such as red,[0.3,0,0],0.3")
   print Util.formatArgument("-left value", "Left boundary location for saved figure [range 0-1]")
   print Util.formatArgument("-leg titles", "Comma-separated list of legend titles. Use '_' to represent space.")
   print Util.formatArgument("-legfs size", "Font size for legend. Set to 0 to hide legend.")
   print Util.formatArgument("-legloc loc", "Where should the legend be placed?  Locations such as 'best', 'upper_left', 'lower_right', 'center'. Use underscore when using two words.")
   print Util.formatArgument("-lw width", "How wide should lines be?")
   print Util.formatArgument("-logx", "Use a logarithmic x-axis")
   print Util.formatArgument("-logy", "Use a logarithmic y-axis")
   print Util.formatArgument("-majlth length", "Length of major tick marks")
   print Util.formatArgument("-majtwid width", "Adjust the thickness of the major tick marks")
   print Util.formatArgument("-minlth length", "Length of minor tick marks")
   print Util.formatArgument("-ms size", "How big should markers be?")
   print Util.formatArgument("-nomargin", "Remove margins (whitespace) in the plot not x[i] <= T.")
   print Util.formatArgument("-right value", "Right boundary location for saved figure [range 0-1]")
   print Util.formatArgument("-sat", "Show satellite image if plotting a map (slow)")
   print Util.formatArgument("-simple", "Make a simpler plot, without extra lines, subplots, etc.")
   print Util.formatArgument("-sp", "Show a line indicating the perfect score")
   print Util.formatArgument("-tickfs size", "Font size for axis ticks")
   print Util.formatArgument("-title text", "Custom title to chart top")
   print Util.formatArgument("-top value", "Top boundary location for saved figure [range 0-1]")
   print Util.formatArgument("-type type", "One of 'plot' (default), 'text', 'map', or 'maprank'.")
   print Util.formatArgument("-xlabel text", "Custom x-axis label")
   print Util.formatArgument("-xlim limits", "Force x-axis limits to the two values lower,upper")
   print Util.formatArgument("-xrot value", "Rotation angle for x-axis labels")
   print Util.formatArgument("-ylabel text", "Custom y-axis label")
   print Util.formatArgument("-ylim limits", "Force y-axis limits to the two values lower,upper")
   print ""
   metrics = Metric.getAllMetrics()
   outputs = Output.getAllOutputs()
   print Util.green("Metrics (-m):")
   print "  (For a full description, run verif -m <metric>)"
   metricOutputs = metrics + outputs
   metricOutputs.sort(key=lambda x: x[0].lower(), reverse=False)
   for m in metricOutputs:
      name = m[0].lower()
      if(m[1].isValid()):
         desc = m[1].summary()
         print Util.formatArgument(name, desc)
         # print "   %-14s%s" % (name, textwrap.fill(desc, 80).replace('\n', '\n                 ')),
         # print ""
   if(data is not None):
      print ""
      print "  Or one of the following, which plots the raw score from the file:"
      print " ",
      metrics = data.getMetrics()
      for metric in metrics:
         print metric,
   print ""
   print ""
   print Util.green("File formats:")
   print Input.Text.description()
   print Input.Comps.description()

if __name__ == '__main__':
       main()
