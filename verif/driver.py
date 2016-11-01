import sys
import os
import verif.axis
import verif.data
import verif.input
import verif.metric
import verif.output
import verif.util
import verif.version
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
   lat_range = None
   lon_range = None
   elev_range = None
   thresholds = None
   clim_file = None
   clim_type = "subtract"
   leg = None
   ylabel = None
   xlabel = None
   title = None
   times = None
   offsets = None
   axis = None
   sdim = None
   figsize = None
   dpi = 100
   no_margin = False
   bin_type = None
   simple = None
   marker_size = None
   line_width = None
   line_colors = None
   tick_font_size = None
   lab_font_size = None
   leg_font_size = None
   title_font_size = None
   leg_loc = None
   type = "plot"
   xrot = None
   major_length = None
   minor_length = None
   major_width = None
   bottom_padding = None
   top_padding = None
   right_padding = None
   lef_padding = None
   Pad = None
   show_perfect = None
   aggregator_name = "mean"
   do_hist = False
   do_sort = False
   do_acc = False
   xlim = None
   ylim = None
   clim = None
   xticks = None
   xticklabels = None
   yticks = None
   yticklabels = None
   version = None
   list_thresholds = False
   list_quantiles = False
   list_locations = False
   list_times = False
   map_type = None
   log_x = False
   log_y = False
   cmap = None
   obs_field = verif.field.Obs()
   fcst_field = verif.field.Fcst()

   # Read command line arguments
   i = 1
   while(i < len(argv)):
      arg = argv[i]
      if(arg[0] == '-'):
         # Process option
         if(arg == "-nomargin"):
            no_margin = True
         elif(arg == "--version"):
            version = True
         elif(arg == "--list-thresholds"):
            list_thresholds = True
         elif(arg == "--list-quantiles"):
            list_quantiles = True
         elif(arg == "--list-locations"):
            list_locations = True
         elif(arg == "--list-times"):
            list_times = True
         elif(arg == "-sp"):
            show_perfect = True
         elif(arg == "-hist"):
            do_hist = True
         elif(arg == "-acc"):
            do_acc = True
         elif(arg == "-sort"):
            do_sort = True
         elif(arg == "-simple"):
            simple = True
         elif(arg == "-logx"):
            log_x = True
         elif(arg == "-logy"):
            log_y = True
         else:
            if(arg == "-f"):
               ofile = argv[i + 1]
            elif(arg == "-l"):
               locations = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-latrange"):
               lat_range = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-lonrange"):
               lon_range = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-elevrange"):
               elev_range = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-x"):
               axisname = argv[i + 1]
               axis = verif.axis.get(axisname)
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
               bin_type = argv[i + 1]
            elif(arg == "-type"):
               type = argv[i + 1]
            elif(arg == "-fs"):
               figsize = argv[i + 1]
            elif(arg == "-dpi"):
               dpi = int(argv[i + 1])
            elif(arg == "-d"):
               dates = verif.util.parse_numbers(argv[i + 1], True)
               times = [verif.util.date_to_unixtime(date) for date in dates]
            elif(arg == "-t"):
               times = verif.util.parse_numbers(argv[i + 1], True)
            elif(arg == "-c"):
               clim_file = argv[i + 1]
               clim_type = "subtract"
            elif(arg == "-C"):
               clim_file = argv[i + 1]
               clim_type = "divide"
            elif(arg == "-xlim"):
               xlim = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-ylim"):
               ylim = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-clim"):
               clim = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-xticks"):
               xticks = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-xticklabels"):
               xticklabels = (argv[i + 1]).split(',')
            elif(arg == "-yticks"):
               yticks = verif.util.parse_numbers(argv[i + 1])
            elif(arg == "-yticklabels"):
               yticklabels = (argv[i + 1]).split(',')
            elif(arg == "-s"):
               sdim = argv[i + 1]
            elif(arg == "-agg"):
               aggregator_name = argv[i + 1]
            elif(arg == "-r"):
               thresholds = np.array(verif.util.parse_numbers(argv[i + 1]))
            elif(arg == "-ms"):
               marker_size = float(argv[i + 1])
            elif(arg == "-lw"):
               line_width = float(argv[i + 1])
            elif(arg == "-lc"):
               line_colors = argv[i + 1]
            elif(arg == "-tickfs"):
               tick_font_size = float(argv[i + 1])
            elif(arg == "-labfs"):
               lab_font_size = float(argv[i + 1])
            elif(arg == "-legfs"):
               leg_font_size = float(argv[i + 1])
            elif(arg == "-legloc"):
               leg_loc = argv[i + 1].replace('_', ' ')
            elif(arg == "-xrot"):
               xrot = float(argv[i + 1])
            elif(arg == "-majlth"):
               major_length = float(argv[i + 1])
            elif(arg == "-minlth"):
               minor_length = float(argv[i + 1])
            elif(arg == "-majwid"):
               major_width = float(argv[i + 1])
            elif(arg == "-bot"):
               bottom_padding = float(argv[i + 1])
            elif(arg == "-top"):
               top_padding = float(argv[i + 1])
            elif(arg == "-right"):
               right_padding = float(argv[i + 1])
            elif(arg == "-left"):
               lef_padding = float(argv[i + 1])
            elif(arg == "-pad"):
               Pad = argv[i + 1]
            elif(arg == "-titlefs"):
               title_font_size = float(argv[i + 1])
            elif(arg == "-cmap"):
               cmap = argv[i + 1]
            elif(arg == "-maptype"):
               map_type = argv[i + 1]
            elif(arg == "-obs"):
               obs_field = verif.field.get(argv[i + 1])
            elif(arg == "-fcst"):
               fcst_field = verif.field.get(argv[i + 1])
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

   if(lat_range is not None and len(lat_range) != 2):
      verif.util.error("-lat_range <values> must have exactly 2 values")

   if(lon_range is not None and len(lon_range) != 2):
      verif.util.error("-lon_range <values> must have exactly 2 values")

   if(elev_range is not None and len(elev_range) != 2):
      verif.util.error("-elev_range <values> must have exactly 2 values")

   if(len(ifiles) > 0):
      inputs = [verif.input.get_input(filename) for filename in ifiles]
      data = verif.data.Data(inputs, clim=clim_file, clim_type=clim_type,
            times=times, offsets=offsets, locations=locations,
            lat_range=lat_range, lon_range=lon_range, elev_range=elev_range,
            legend=leg, obs_field=obs_field, fcst_field=fcst_field)
   else:
      data = None

   if(list_thresholds or list_quantiles or list_locations or list_times):
      if(len(ifiles) == 0):
         verif.util.error("Files are required in order to list thresholds or quantiles")
      if(list_thresholds):
         print "Thresholds:",
         for threshold in data.thresholds:
            print "%g" % threshold,
         print ""
      if(list_quantiles):
         print "Quantiles:",
         for quantile in data.quantiles:
            print "%g" % quantile,
         print ""
      if(list_locations):
         print "    id     lat     lon    elev"
         for location in data.locations:
            print "%6d %7.2f %7.2f %7.1f" % (location.id, location.lat,
                  location.lon, location.elev)
         print ""
      if(list_times):
         for time in data.times:
            print "%d" % time
         print ""
      return
   elif(len(ifiles) == 0 and metric is not None):
      m = verif.metric.get(metric)
      if(m is not None):
         print m.help()
      else:
         m = verif.output.get(metric)
         if(m is not None):
            print m.help()
      return
   elif(len(argv) == 1 or len(ifiles) == 0 or metric is None):
      show_description(data)
      return

   if(figsize is not None):
      figsize = figsize.split(',')
      if(len(figsize) != 2):
         print "-fs figsize must be in the form: width,height"
         sys.exit(1)

   m = None

   # Handle special plots
   if(metric == "pithist"):
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
   elif(metric == "impact"):
      pl = verif.output.Impact()
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
      m = verif.metric.get(metric)
      if(m is None):
         verif.util.error("Unknown plot: %s" % metric)

      if aggregator_name is not None:
         m.aggregator = verif.aggregator.get(aggregator_name)

      # Output type
      if(type in ["plot", "text", "csv", "map", "maprank"]):
         pl = verif.output.Standard(m)
         if do_sort:
            pl = verif.output.Sort(m)
         if do_hist:
            pl = verif.output.Hist(m)
         pl.show_acc = do_acc
      else:
         verif.util.error("Type not understood")

   # Rest dimension of '-x' is not allowed
   if(axis is not None and not pl.supports_x):
      verif.util.warning(metric + " does not support -x. Ignoring it.")
      axis = None

   # Reset dimension if 'threshold' is not allowed
   if(axis == verif.axis.Threshold() and
         ((not pl.supports_threshold) or (m is not None and not m.supports_threshold))):
      verif.util.warning(metric + " does not support '-x threshold'. Ignoring it.")
      thresholds = None
      axis = None

   # Create thresholds if needed
   if((thresholds is None) and (pl.requires_threshold or
         (m is not None and m.requires_threshold))):
      obs = data.get_scores(verif.field.Obs(), 0)[0]
      fcst = data.get_scores(verif.field.Fcst(), 0)[0]
      smin = min(np.nanmin(obs), np.nanmin(fcst))
      smax = max(np.nanmax(obs), np.nanmax(fcst))
      thresholds = np.linspace(smin, smax, 10)
      verif.util.warning("Missing '-r <thresholds>'. Automatically setting thresholds.")

   # Set plot parameters
   if(simple is not None):
      pl.simple = simple
   if(marker_size is not None):
      pl.ms(marker_size)
   if(line_width is not None):
      pl.lw(line_width)
   if(line_colors is not None):
      pl.line_colors = line_colors
   if(lab_font_size is not None):
      pl.lab_font_size = lab_font_size
   if(leg_font_size is not None):
      pl.leg_font_size = leg_font_size
   if(title_font_size is not None):
      pl.title_font_size = title_font_size
   if(leg_loc is not None):
      pl.leg_loc = leg_loc
   if(tick_font_size is not None):
      pl.tick_font_size = tick_font_size
   if(xrot is not None):
      pl.xrot = xrot
   if(major_length is not None):
      pl.major_length = major_length
   if(minor_length is not None):
      pl.minor_length = minor_length
   if(major_width is not None):
      pl.major_width = major_width
   if(bottom_padding is not None):
      pl.bottom = bottom_padding
   if(top_padding is not None):
      pl.top = top_padding
   if(right_padding is not None):
      pl.right = right_padding
   if(lef_padding is not None):
      pl.left = lef_padding
   if(Pad is not None):
      pl.pad = None
   if(bin_type is not None):
      pl.bin_type = bin_type
   if(show_perfect is not None):
      pl.show_perfect = show_perfect
   if(xlim is not None):
      pl.xlim(xlim)
   if(ylim is not None):
      pl.ylim(ylim)
   if(clim is not None):
      pl.clim(clim)
   if(xticks is not None):
      pl.xticks(xticks)
   if(xticklabels is not None):
      pl.xticklabels(xticklabels)
   if(yticks is not None):
      pl.yticks(yticks)
   if(yticklabels is not None):
      pl.yticklabels(yticklabels)
   if(log_x is not None):
      pl.log_x = log_x
   if(log_y is not None):
      pl.log_y = log_y
   if(cmap is not None):
      pl.cmap = cmap
   if(map_type is not None):
      pl.map_type = map_type
   pl.filename = ofile
   if thresholds is not None:
      pl.thresholds = thresholds
   pl.figsize = figsize
   pl.dpi = dpi
   if axis is not None:
      pl.axis = axis
   pl.aggregator = verif.aggregator.get(aggregator_name)
   pl.show_margin = not no_margin
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
      pl.show_rank = True
      pl.map(data)
   else:
      pl.plot(data)


def get_aggregation_string():
   aggregators = verif.aggregator.get_all()
   value = "Aggregation type: "
   for aggregator in aggregators:
      if aggregator.name() != "quantile":
         value = "%s'%s', " % (value, aggregator.name())
   value = value + "or a number between 0 and 1. Some metrics computes a value for each value on the x-axis. Which function should be used to do the collapsing? Default is 'mean'. 'meanabs' is the mean absolute value. Only supported by some metrics. A number between 0 and 1 returns a specific quantile (e.g.  0.5 is the median)"
   return value


def show_description(data=None):
   desc = "Program to compute verification scores for weather forecasts. Can be " \
          "used to compare forecasts from different files. In that case only times, "\
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
   print verif.util.format_argument("--list-times", "What times are available in the files?")
   print verif.util.format_argument("--list-locations", "What locations are available in the files?")
   print verif.util.format_argument("--list-quantiles", "What quantiles are available in the files?")
   print verif.util.format_argument("--list-thresholds", "What thresholds are available in the files?")
   print verif.util.format_argument("--version", "What version of verif is this?")
   print ""
   print verif.util.green("Options:")
   print "Note: vectors can be entered using commas, or MATLAB syntax (i.e 3:5 is 3,4,5 and 3:2:7 is 3,5,7)"
   # Dimensions
   print verif.util.green("  Dimensions and subset:")
   print verif.util.format_argument("-elevrange range", "Limit the verification to locations within minelev,maxelev.")
   print verif.util.format_argument("-d dates", "A vector of dates in YYYYMMDD format, e.g.  20130101:20130201.")
   print verif.util.format_argument("-t times", "A vector of unix timestamps.")
   print verif.util.format_argument("-fcst", "Which field should be used as the forecast?")
   print verif.util.format_argument("-l locations", "Limit the verification to these location IDs.")
   print verif.util.format_argument("-latrange range", "Limit the verification to locations within minlat,maxlat.")
   print verif.util.format_argument("-lonrange range", "Limit the verification to locations within minlon,maxlon.")
   print verif.util.format_argument("-o offsets", "Limit the verification to these offsets (in hours).")
   print verif.util.format_argument("-obs", "Which field should be used as the observation?")
   print verif.util.format_argument("-r thresholds", "Compute scores for these thresholds (only used by some metrics).")
   print verif.util.format_argument("-x dim", "Plot this dimension on the x-axis: date, offset, year, month, location, locationId, elev, lat, lon, threshold, or none. Not supported by all metrics. If not specified, then a default is used based on the metric. 'none' collapses all dimensions and computes one value.")

   # Data manipulation
   print verif.util.green("  Data manipulation:")
   print verif.util.format_argument("-acc", "Plot accumulated values. Only works for non-derived metrics")
   print verif.util.format_argument("-agg type", get_aggregation_string())
   print verif.util.format_argument("-b type", "One of 'below', 'within', or 'above'. For threshold plots (ets, hit, within, etc) 'below/above' computes frequency below/above the threshold, and 'within' computes the frequency between consecutive thresholds.")
   print verif.util.format_argument("-c file", "File containing climatology data. Subtract all forecasts and obs with climatology values.")
   print verif.util.format_argument("-C file", "File containing climatology data. Divide all forecasts and obs by climatology values.")
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
   print verif.util.format_argument("-xticklabels labels", "A comma-separated list of labels for the x-axis ticks")
   print verif.util.format_argument("-xrot value", "Rotation angle for x-axis labels")
   print verif.util.format_argument("-ylabel text", "Custom y-axis label")
   print verif.util.format_argument("-ylim limits", "Force y-axis limits to the two values lower,upper")
   print verif.util.format_argument("-yticks ticks", "A vector of values to put ticks on the y-axis")
   print verif.util.format_argument("-xticklabels labels", "A comma-separated list of labels for the y-axis ticks")
   print ""
   metrics = verif.metric.get_all()
   outputs = verif.output.get_all()
   print verif.util.green("Metrics (-m):")
   print "  (For a full description, run verif -m <metric>)"
   metric_outputs = metrics + outputs
   metric_outputs.sort(key=lambda x: x[0].lower(), reverse=False)
   for m in metric_outputs:
      name = m[0].lower()
      if(m[1].is_valid()):
         desc = m[1].summary()
         print verif.util.format_argument(name, desc)
         # print "   %-14s%s" % (name, textwrap.fill(desc, 80).replace('\n', '\n                 ')),
         # print ""
   print ""
   print ""
   print verif.util.green("File formats:")
   print verif.input.Text.description
   print verif.input.Comps.description

if __name__ == '__main__':
       main()
