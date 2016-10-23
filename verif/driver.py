import sys
import os
import verif.data
import verif.input
import verif.metric
import verif.output
import verif.util
import verif.version
import matplotlib.pyplot as mpl
import textwrap
import numpy as np


def run(argv):
   ############
   # Defaults #
   ############
   ifiles = list()
   ofile = None
   metric = None
   locations = None
   latRange = None
   lonRange = None
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
   titleFontSize = None
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
   xticks = None
   yticks = None
   version = None
   listThresholds = False
   listQuantiles = False
   listLocations = False
   listDates = False
   mapType = None
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
         else:
            if(arg == "-f"):
               ofile = argv[i + 1]
            elif(arg == "-l"):
               locations = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-latrange"):
               latRange = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-lonrange"):
               lonRange = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-elevrange"):
               elevRange = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-t"):
               training = int(argv[i + 1])
            elif(arg == "-x"):
               xdim = argv[i + 1]
            elif(arg == "-o"):
               offsets = verif.util.parse_numbers(argv[i + 1])
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
               dates = verif.util.parse_numbers(argv[i + 1], True)
            elif(arg == "-c"):
               climFile = argv[i + 1]
               climType = "subtract"
            elif(arg == "-C"):
               climFile = argv[i + 1]
               climType = "divide"
            elif(arg == "-xlim"):
               xlim = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-ylim"):
               ylim = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-clim"):
               clim = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-xticks"):
               xticks = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-yticks"):
               yticks = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-s"):
               sdim = argv[i + 1]
            elif(arg == "-ct"):
               aggregatorName = argv[i + 1]
            elif(arg == "-r"):
               thresholds = verif.util.parse_numbers(argv[i + 1])
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
            elif(arg == "-titlefs"):
               titleFontSize = float(argv[i + 1])
            elif(arg == "-cmap"):
               cmap = argv[i + 1]
            elif(arg == "-maptype"):
               mapType = argv[i + 1]
            elif(arg == "-m"):
               metric = argv[i + 1]
            else:
               verif.util.error("Flag '" + argv[i] + "' not recognized")
            i = i + 1
      else:
         ifiles.append(argv[i])
      i = i + 1

   if(version):
      print "Version: " + verif.version.__version__
      return

   # Deal with legend entries
   if(leg is not None):
      leg = leg.split(',')
      for i in range(0, len(leg)):
         leg[i] = leg[i].replace('_', ' ')

   if(latRange is not None and len(latRange) != 2):
      verif.util.error("-latRange <values> must have exactly 2 values")

   if(lonRange is not None and len(lonRange) != 2):
      verif.util.error("-lonRange <values> must have exactly 2 values")

   if(elevRange is not None and len(elevRange) != 2):
      verif.util.error("-elevRange <values> must have exactly 2 values")

   if(len(ifiles) > 0):
      data = verif.data.Data(ifiles, clim=climFile, climType=climType, dates=dates,
            offsets=offsets, locations=locations, latRange=latRange, lonRange=lonRange,
            elevRange=elevRange, training=training, legend=leg)
   else:
      data = None

   if(listThresholds or listQuantiles or listLocations or listDates):
      if(len(ifiles) == 0):
         verif.util.error("Files are required in order to list thresholds or quantiles")
      if(listThresholds):
         print "Thresholds:",
         for threshold in data.get_thresholds():
            print "%g" % threshold,
         print ""
      if(listQuantiles):
         print "Quantiles:",
         for quantile in data.get_quantiles():
            print "%g" % quantile,
         print ""
      if(listLocations):
         print "    id     lat     lon    elev"
         for station in data.get_stations():
            print "%6d %7.2f %7.2f %7.1f" % (station.id(), station.lat(),
                  station.lon(), station.elev())
         print ""
      if(listDates):
         dates = data.get_axis_values("date")
         dates = verif.util.convert_to_yyyymmdd(dates)
         for date in dates:
            print "%d" % date
         print ""
      return
   elif(len(ifiles) == 0 and metric is not None):
      m = verif.metric.get_metric(metric)
      if(m is not None):
         print m.help()
      else:
         m = verif.output.get_output(metric)
         if(m is not None):
            print m.help()
      return
   elif(len(argv) == 1 or len(ifiles) == 0 or metric is None):
      show_description(data)
      return

   if(figSize is not None):
      figSize = figSize.split(',')
      if(len(figSize) != 2):
         print "-fs figSize must be in the form: width,height"
         sys.exit(1)

   m = None

   # Handle special plots
   if(doHist):
      pl = verif.output.Hist(metric)
   elif(doSort):
      pl = verif.output.Sort(metric)
   elif(metric == "pithist"):
      m = verif.metric.Pit("pit")
      pl = verif.output.PitHist(m)
   elif(metric == "obsfcst"):
      pl = verif.output.ObsFcst()
   elif(metric == "timeseries"):
      pl = verif.output.TimeSeries()
   elif(metric == "meteo"):
      pl = verif.output.Meteo()
   elif(metric == "qq"):
      pl = verif.output.QQ()
   elif(metric == "cond"):
      pl = verif.output.Cond()
   elif(metric == "against"):
      pl = verif.output.Against()
   elif(metric == "count"):
      pl = verif.output.Count()
   elif(metric == "scatter"):
      pl = verif.output.Scatter()
   elif(metric == "change"):
      pl = verif.output.Change()
   elif(metric == "spreadskill"):
      pl = verif.output.SpreadSkill()
   elif(metric == "taylor"):
      pl = verif.output.Taylor()
   elif(metric == "error"):
      pl = verif.output.Error()
   elif(metric == "freq"):
      pl = verif.output.Freq()
   elif(metric == "roc"):
      pl = verif.output.Roc()
   elif(metric == "droc"):
      pl = verif.output.DRoc()
   elif(metric == "droc0"):
      pl = verif.output.DRoc0()
   elif(metric == "drocnorm"):
      pl = verif.output.DRocNorm()
   elif(metric == "reliability"):
      pl = verif.output.Reliability()
   elif(metric == "discrimination"):
      pl = verif.output.Discrimination()
   elif(metric == "performance"):
      pl = verif.output.Performance()
   elif(metric == "invreliability"):
      pl = verif.output.InvReliability()
   elif(metric == "igncontrib"):
      pl = verif.output.IgnContrib()
   elif(metric == "economicvalue"):
      pl = verif.output.EconomicValue()
   elif(metric == "marginal"):
      pl = verif.output.Marginal()
   else:
      # Standard plots
      # Attempt at automating
      m = verif.metric.get_metric(metric)
      if(m is None):
         m = verif.metric.Default(metric)

      m.set_aggregator(aggregatorName)

      # Output type
      if(type in ["plot", "text", "csv", "map", "maprank"]):
         pl = verif.output.Default(m)
         pl.set_show_acc(doAcc)
      else:
         verif.util.error("Type not understood")

   # Rest dimension of '-x' is not allowed
   if(xdim is not None and not pl.supportsX()):
      verif.util.warning(metric + " does not support -x. Ignoring it.")
      xdim = None

   # Reset dimension if 'threshold' is not allowed
   if(xdim == "threshold" and
         ((not pl.supports_threshold()) or (m is not None and not m.supports_threshold()))):
      verif.util.warning(metric + " does not support '-x threshold'. Ignoring it.")
      thresholds = None
      xdim = None

   # Create thresholds if needed
   if((thresholds is None) and (pl.requires_thresholds() or
         (m is not None and m.requires_thresholds()))):
      data.set_axis("none")
      obs = data.get_scores("obs")[0]
      fcst = data.get_scores("fcst")[0]
      smin = min(min(obs), min(fcst))
      smax = max(max(obs), max(fcst))
      thresholds = np.linspace(smin, smax, 10)
      verif.util.warning("Missing '-r <thresholds>'. Automatically setting\
            thresholds.")

   # Set plot parameters
   if(simple is not None):
      pl.set_simple(simple)
   if(markerSize is not None):
      pl.ms(markerSize)
   if(lineWidth is not None):
      pl.lw(lineWidth)
   if(lineColors is not None):
      pl.set_line_colors(lineColors)
   if(labFontSize is not None):
      pl.set_lab_font_size(labFontSize)
   if(legFontSize is not None):
      pl.set_leg_font_size(legFontSize)
   if(titleFontSize is not None):
      pl.set_title_font_size(titleFontSize)
   if(legLoc is not None):
      pl.set_leg_loc(legLoc)
   if(tickFontSize is not None):
      pl.set_tick_font_size(tickFontSize)
   if(XRotation is not None):
      pl.set_x_rotation(XRotation)
   if(MajorLength is not None):
      pl.set_major_length(MajorLength)
   if(MinorLength is not None):
      pl.set_minor_length(MinorLength)
   if(MajorWidth is not None):
      pl.set_major_width(MajorWidth)
   if(Bottom is not None):
      pl.set_bottom(Bottom)
   if(Top is not None):
      pl.set_top(Top)
   if(Right is not None):
      pl.set_right(Right)
   if(Left is not None):
      pl.set_left(Left)
   if(Pad is not None):
      pl.set_pad(None)
   if(binType is not None):
      pl.set_bin_type(binType)
   if(showPerfect is not None):
      pl.set_show_perfect(showPerfect)
   if(xlim is not None):
      pl.xlim(xlim)
   if(ylim is not None):
      pl.set_y_lim(ylim)
   if(clim is not None):
      pl.clim(clim)
   if(xticks is not None):
      pl.xticks(xticks)
   if(yticks is not None):
      pl.yticks(yticks)
   if(logX is not None):
      pl.set_log_x(logX)
   if(logY is not None):
      pl.set_log_y(logY)
   if(cmap is not None):
      pl.set_cmap(cmap)
   if(mapType is not None):
      pl.set_map_type(mapType)
   pl.set_filename(ofile)
   pl.set_thresholds(thresholds)
   pl.set_fig_size(figSize)
   pl.set_dpi(dpi)
   pl.set_axis(xdim)
   pl.set_aggregator_name(aggregatorName)
   pl.set_show_margin(not noMargin)
   pl.ylabel(ylabel)
   pl.xlabel(xlabel)
   pl.title(title)

   if(type == "text"):
      pl.text(data)
   elif(type == "csv"):
      pl.csv(data)
   elif(type == "map"):
      pl.map(data)
   elif(type == "maprank"):
      pl.set_show_rank(True)
      pl.map(data)
   else:
      pl.plot(data)


def show_description(data=None):
   desc = "Program to compute verification scores for weather forecasts. Can be " \
          "used to compare forecasts from different files. In that case only dates, "\
          "offsets, and locations that are common to all forecast files are used."
   print textwrap.fill(desc, verif.util.get_text_width())
   print ""
   print "usage: verif files -m metric [options]"
   print "       verif files [--list-thresholds] [--list-quantiles] [--list-locations]"
   print "       verif --version"
   print ""
   print verif.util.green("Arguments:")
   print verif.util.format_argument("files", "One or more verification files in NetCDF or text format (see 'File Formats' below).")
   print verif.util.format_argument("-m metric", "Which verification metric to use? See 'Metrics' below.")
   print verif.util.format_argument("--list-thresholds", "What thresholds are available in the files?")
   print verif.util.format_argument("--list-quantiles", "What quantiles are available in the files?")
   print verif.util.format_argument("--list-locations", "What locations are available in the files?")
   print verif.util.format_argument("--version", "What version of verif is this?")
   print ""
   print verif.util.green("Options:")
   print "Note: vectors can be entered using commas, or MATLAB syntax (i.e 3:5 is 3,4,5 and 3:2:7 is 3,5,7)"
   # Dimensions
   print verif.util.green("  Dimensions and subset:")
   print verif.util.format_argument("-elevrange range", "Limit the verification to locations within minelev,maxelev.")
   print verif.util.format_argument("-d dates", "A vector of dates in YYYYMMDD format, e.g.  20130101:20130201.")
   print verif.util.format_argument("-l locations", "Limit the verification to these location IDs.")
   print verif.util.format_argument("-latrange range", "Limit the verification to locations within minlat,maxlat.")
   print verif.util.format_argument("-lonrange range", "Limit the verification to locations within minlon,maxlon.")
   print verif.util.format_argument("-o offsets", "Limit the verification to these offsets (in hours).")
   print verif.util.format_argument("-r thresholds", "Compute scores for these thresholds (only used by some metrics).")
   print verif.util.format_argument("-t period", "Allow this many days of training, i.e. remove this many days from the beginning of the verification.")
   print verif.util.format_argument("-x dim", "Plot this dimension on the x-axis: date, offset, year, month, location, locationId, elev, lat, lon, threshold, or none. Not supported by all metrics. If not specified, then a default is used based on the metric. 'none' collapses all dimensions and computes one value.")

   # Data manipulation
   print verif.util.green("  Data manipulation:")
   print verif.util.format_argument("-acc", "Plot accumulated values. Only works for non-derived metrics")
   print verif.util.format_argument("-b type", "One of 'below', 'within', or 'above'. For threshold plots (ets, hit, within, etc) 'below/above' computes frequency below/above the threshold, and 'within' computes the frequency between consecutive thresholds.")
   print verif.util.format_argument("-c file", "File containing climatology data. Subtract all forecasts and obs with climatology values.")
   print verif.util.format_argument("-C file", "File containing climatology data. Divide all forecasts and obs by climatology values.")
   print verif.util.format_argument("-ct type", "Collapsing type: 'count', 'min', 'mean', 'meanabs', 'median', 'max', 'range', 'std', 'sum', or a number between 0 and 1. Some metrics computes a value for each value on the x-axis. Which function should be used to do the collapsing? Default is 'mean'. 'meanabs' is the mean absolute value. Only supported by some metrics. A number between 0 and 1 returns a specific quantile (e.g.  0.5 is the median)")
   print verif.util.format_argument("-hist", "Plot values as histogram. Only works for non-derived metrics")
   print verif.util.format_argument("-sort", "Plot values sorted. Only works for non-derived metrics")

   # Plot options
   print verif.util.green("  Plotting options:")
   print verif.util.format_argument("-bot value", "Bottom boundary location for saved figure [range 0-1]")
   print verif.util.format_argument("-clim limits", "Force colorbar limits to the two values lower,upper")
   print verif.util.format_argument("-cmap colormap", "Use this colormap when possible (e.g. jet, inferno, RdBu)")
   print verif.util.format_argument("-dpi value", "Resolution of image in dots per inch (default 100)")
   print verif.util.format_argument("-f file", "Save image to this filename")
   print verif.util.format_argument("-fs size", "Set figure size width,height (in inches). Default 8x6.")
   print verif.util.format_argument("-labfs size", "Font size for axis labels")
   print verif.util.format_argument("-lc colors", "Comma-separated list of line colors, such as red,[0.3,0,0],0.3")
   print verif.util.format_argument("-left value", "Left boundary location for saved figure [range 0-1]")
   print verif.util.format_argument("-leg titles", "Comma-separated list of legend titles. Use '_' to represent space.")
   print verif.util.format_argument("-legfs size", "Font size for legend. Set to 0 to hide legend.")
   print verif.util.format_argument("-legloc loc", "Where should the legend be placed?  Locations such as 'best', 'upper_left', 'lower_right', 'center'. Use underscore when using two words.")
   print verif.util.format_argument("-lw width", "How wide should lines be?")
   print verif.util.format_argument("-logx", "Use a logarithmic x-axis")
   print verif.util.format_argument("-logy", "Use a logarithmic y-axis")
   print verif.util.format_argument("-majlth length", "Length of major tick marks")
   print verif.util.format_argument("-majtwid width", "Adjust the thickness of the major tick marks")
   print verif.util.format_argument("-maptype", "One of 'simple', 'sat', 'topo', or any of these http://server.arcgisonline.com/arcgis/rest/services names.  'simple' shows a basic ocean/lakes/land map, 'sat' shows a satellite image, and 'topo' a topographical map. Only relevant when '-type map' has been selected.")
   print verif.util.format_argument("-minlth length", "Length of minor tick marks")
   print verif.util.format_argument("-ms size", "How big should markers be?")
   print verif.util.format_argument("-nomargin", "Remove margins (whitespace) in the plot not x[i] <= T.")
   print verif.util.format_argument("-right value", "Right boundary location for saved figure [range 0-1]")
   print verif.util.format_argument("-simple", "Make a simpler plot, without extra lines, subplots, etc.")
   print verif.util.format_argument("-sp", "Show a line indicating the perfect score")
   print verif.util.format_argument("-tickfs size", "Font size for axis ticks")
   print verif.util.format_argument("-titlefs size", "Font size for title.")
   print verif.util.format_argument("-title text", "Custom title to chart top")
   print verif.util.format_argument("-top value", "Top boundary location for saved figure [range 0-1]")
   print verif.util.format_argument("-type type", "One of 'plot' (default), 'text', 'csv', 'map', or 'maprank'.")
   print verif.util.format_argument("-xlabel text", "Custom x-axis label")
   print verif.util.format_argument("-xlim limits", "Force x-axis limits to the two values lower,upper")
   print verif.util.format_argument("-xticks ticks", "A vector of values to put ticks on the x-axis")
   print verif.util.format_argument("-xrot value", "Rotation angle for x-axis labels")
   print verif.util.format_argument("-ylabel text", "Custom y-axis label")
   print verif.util.format_argument("-ylim limits", "Force y-axis limits to the two values lower,upper")
   print verif.util.format_argument("-yticks ticks", "A vector of values to put ticks on the y-axis")
   print ""
   metrics = verif.metric.get_all()
   outputs = verif.output.get_all()
   print verif.util.green("Metrics (-m):")
   print "  (For a full description, run verif -m <metric>)"
   metricOutputs = metrics + outputs
   metricOutputs.sort(key=lambda x: x[0].lower(), reverse=False)
   for m in metricOutputs:
      name = m[0].lower()
      if(m[1].is_valid()):
         desc = m[1].summary()
         print verif.util.format_argument(name, desc)
         # print "   %-14s%s" % (name, textwrap.fill(desc, 80).replace('\n', '\n                 ')),
         # print ""
   if(data is not None):
      print ""
      print "  Or one of the following, which plots the raw score from the file:"
      print " ",
      metrics = data.get_metrics()
      for metric in metrics:
         print metric,
   print ""
   print ""
   print verif.util.green("File formats:")
   print verif.input.Text.description()
   print verif.input.Comps.description()

if __name__ == '__main__':
       main()
