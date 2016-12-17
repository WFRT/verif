import sys
import os
import verif.axis
import verif.data
import verif.input
import verif.metric
import verif.metric_type
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
   quantiles = None
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
   plot_type = "plot"
   grid = True
   xrot = None
   yrot = None
   bottom_padding = None
   top_padding = None
   right_padding = None
   left_padding = None
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
         elif(arg == "-nogrid"):
            grid = False
         else:
            if len(argv) <= i + 1:
               verif.util.error("Missing value after %s" % argv[i])
            arg_next = argv[i + 1]
            if(arg == "-f"):
               ofile = arg_next
            elif(arg == "-l"):
               locations = verif.util.parse_numbers(arg_next)
            elif(arg == "-latrange"):
               lat_range = verif.util.parse_numbers(arg_next)
            elif(arg == "-lonrange"):
               lon_range = verif.util.parse_numbers(arg_next)
            elif(arg == "-elevrange"):
               elev_range = verif.util.parse_numbers(arg_next)
            elif(arg == "-x"):
               axisname = arg_next
               axis = verif.axis.get(axisname)
            elif(arg == "-o"):
               offsets = verif.util.parse_numbers(arg_next)
            elif(arg == "-leg"):
               leg = unicode(arg_next, 'utf8')
            elif(arg == "-ylabel"):
               ylabel = unicode(arg_next, 'utf8')
            elif(arg == "-xlabel"):
               xlabel = unicode(arg_next, 'utf8')
            elif(arg == "-title"):
               title = unicode(arg_next, 'utf8')
            elif(arg == "-b"):
               bin_type = arg_next
            elif(arg == "-type"):
               plot_type = arg_next
            elif(arg == "-fs"):
               figsize = arg_next
            elif(arg == "-dpi"):
               dpi = int(arg_next)
            elif(arg == "-d"):
               dates = verif.util.parse_numbers(arg_next, True)
               times = [verif.util.date_to_unixtime(date) for date in dates]
            elif(arg == "-t"):
               times = verif.util.parse_numbers(arg_next, True)
            elif(arg == "-c"):
               clim_file = arg_next
               clim_type = "subtract"
            elif(arg == "-C"):
               clim_file = arg_next
               clim_type = "divide"
            elif(arg == "-xlim"):
               xlim = verif.util.parse_numbers(arg_next)
            elif(arg == "-ylim"):
               ylim = verif.util.parse_numbers(arg_next)
            elif(arg == "-clim"):
               clim = verif.util.parse_numbers(arg_next)
            elif(arg == "-xticks"):
               xticks = verif.util.parse_numbers(arg_next)
            elif(arg == "-xticklabels"):
               xticklabels = (arg_next).split(',')
            elif(arg == "-yticks"):
               yticks = verif.util.parse_numbers(arg_next)
            elif(arg == "-yticklabels"):
               yticklabels = (arg_next).split(',')
            elif(arg == "-s"):
               sdim = arg_next
            elif(arg == "-agg"):
               aggregator_name = arg_next
            elif(arg == "-r"):
               thresholds = np.array(verif.util.parse_numbers(arg_next))
            elif(arg == "-q"):
               quantiles = np.array(verif.util.parse_numbers(arg_next))
               if np.min(quantiles) < 0 or np.max(quantiles) > 1:
                  verif.util.error("Quantiles must be between 0 and 1 inclusive")
            elif(arg == "-ms"):
               marker_size = float(arg_next)
            elif(arg == "-lw"):
               line_width = float(arg_next)
            elif(arg == "-lc"):
               line_colors = arg_next
            elif(arg == "-tickfs"):
               tick_font_size = float(arg_next)
            elif(arg == "-labfs"):
               lab_font_size = float(arg_next)
            elif(arg == "-legfs"):
               leg_font_size = float(arg_next)
            elif(arg == "-legloc"):
               leg_loc = arg_next.replace('_', ' ')
            elif(arg == "-xrot"):
               xrot = float(arg_next)
            elif(arg == "-yrot"):
               yrot = float(arg_next)
            elif(arg == "-bottom"):
               bottom_padding = float(arg_next)
            elif(arg == "-top"):
               top_padding = float(arg_next)
            elif(arg == "-right"):
               right_padding = float(arg_next)
            elif(arg == "-left"):
               left_padding = float(arg_next)
            elif(arg == "-pad"):
               Pad = arg_next
            elif(arg == "-titlefs"):
               title_font_size = float(arg_next)
            elif(arg == "-cmap"):
               cmap = arg_next
            elif(arg == "-maptype"):
               map_type = arg_next
            elif(arg == "-obs"):
               obs_field = verif.field.get(arg_next)
            elif(arg == "-fcst"):
               fcst_field = verif.field.get(arg_next)
            elif(arg == "-m"):
               metric = arg_next
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
      pl = verif.output.PitHist()
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
      if(plot_type in ["plot", "text", "csv", "map", "maprank"]):
         if do_sort:
            field = verif.field.get(metric)
            pl = verif.output.Sort(field)
         elif do_hist:
            field = verif.field.get(metric)
            pl = verif.output.Hist(field)
         else:
            pl = verif.output.Standard(m)
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
   if thresholds is None:
      type = None
      if pl.require_threshold_type == "deterministic":
         type = "deterministic"
      elif pl.require_threshold_type == "threshold":
         type = "threshold"
      elif pl.require_threshold_type == "probabilistic":
         type = "probabilistic"
      elif m is not None:
         if m.require_threshold_type == "deterministic":
            type = "deterministic"
         elif m.require_threshold_type == "threshold":
            type = "threshold"
         elif m.require_threshold_type == "probabilistic":
            type = "probabilistic"

      if type == "deterministic":
         obs = data.get_scores(verif.field.Obs(), 0)
         fcst = data.get_scores(verif.field.Fcst(), 0)
         smin = min(np.nanmin(obs), np.nanmin(fcst))
         smax = max(np.nanmax(obs), np.nanmax(fcst))
         thresholds = np.linspace(smin, smax, 10)
         verif.util.warning("Missing '-r <thresholds>'. Automatically setting thresholds.")
      elif type == "threshold":
         thresholds = data.thresholds
         verif.util.warning("Missing '-r <thresholds>'. Automatically setting thresholds.")
      elif type == "quantile":
         thresholds = data.quantiles
         verif.util.warning("Missing '-r <thresholds>'. Automatically setting thresholds.")

   # Set plot parameters
   if(simple is not None):
      pl.simple = simple
   if(marker_size is not None):
      pl.ms = marker_size
   if(line_width is not None):
      pl.lw = line_width
   if(line_colors is not None):
      pl.line_colors = line_colors
   if(lab_font_size is not None):
      pl.labfs = lab_font_size
   if(leg_font_size is not None):
      pl.legfs = leg_font_size
   if(title_font_size is not None):
      pl.title_font_size = title_font_size
   if(leg_loc is not None):
      pl.leg_loc = leg_loc
   if(tick_font_size is not None):
      pl.tick_font_size = tick_font_size
   if(xrot is not None):
      pl.xrot = xrot
   if(yrot is not None):
      pl.yrot = yrot
   if(bottom_padding is not None):
      pl.bottom = bottom_padding
   if(top_padding is not None):
      pl.top = top_padding
   if(right_padding is not None):
      pl.right = right_padding
   if(left_padding is not None):
      pl.left = left_padding
   if(Pad is not None):
      pl.pad = None
   if(bin_type is not None):
      pl.bin_type = bin_type
   if(show_perfect is not None):
      pl.show_perfect = show_perfect
   if(xlim is not None):
      pl.xlim = xlim
   if(ylim is not None):
      pl.ylim = ylim
   if(clim is not None):
      pl.clim = clim
   if(xticks is not None):
      pl.xticks = xticks
   if(xticklabels is not None):
      pl.xticklabels = xticklabels
   if(yticks is not None):
      pl.yticks = yticks
   if(yticklabels is not None):
      pl.yticklabels = yticklabels
   if(log_x is not None):
      pl.log_x = log_x
   if(log_y is not None):
      pl.log_y = log_y
   pl.grid = grid
   if(cmap is not None):
      pl.cmap = cmap
   if(map_type is not None):
      pl.map_type = map_type
   pl.filename = ofile
   if thresholds is not None:
      pl.thresholds = thresholds
   if quantiles is not None:
      pl.quantiles = quantiles
   pl.figsize = figsize
   pl.dpi = dpi
   if axis is not None:
      pl.axis = axis
   pl.aggregator = verif.aggregator.get(aggregator_name)
   pl.show_margin = not no_margin
   if ylabel is not None:
      pl.ylabel = ylabel
   if xlabel is not None:
      pl.xlabel = xlabel
   if title is not None:
      pl.title = title

   if(plot_type == "text"):
      pl.text(data)
   elif(plot_type == "csv"):
      pl.csv(data)
   elif(plot_type == "map"):
      pl.map(data)
   elif(plot_type == "maprank"):
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
   value = value + "or a number between 0 and 1. Some metrics computes a value for each value on the x-axis. Which function should be used to do the aggregation? Default is 'mean'. Only supported by some metrics. A number between 0 and 1 returns a specific quantile (e.g. 0.5 is the median)."
   return value


def show_description(data=None):
   desc = "Program to compute verification scores for weather forecasts. Can be " \
          "used to compare forecasts from different files. In that case only times, "\
          "offsets, and locations that are common to all forecast files are used."
   print textwrap.fill(desc, get_text_width())
   print ""
   print "usage: verif files -m metric [options]"
   print "       verif files [--list-thresholds] [--list-quantiles] [--list-locations]"
   print "       verif --version"
   print ""
   print verif.util.green("Arguments:")
   print format_argument("files", "One or more verification files in NetCDF or text format (see 'File Formats' below).")
   print format_argument("-m metric", "Which verification metric to use? See 'Metrics' below.")
   print format_argument("--list-times", "What times are available in the files?")
   print format_argument("--list-locations", "What locations are available in the files?")
   print format_argument("--list-quantiles", "What quantiles are available in the files?")
   print format_argument("--list-thresholds", "What thresholds are available in the files?")
   print format_argument("--version", "What version of verif is this?")
   print ""
   print verif.util.green("Options:")
   print "Note: vectors can be entered using commas, or MATLAB syntax (i.e 3:5 is 3,4,5 and 3:2:7 is 3,5,7)"
   # Dimensions
   print verif.util.green("  Dimensions and subset:")
   print format_argument("-elevrange range", "Limit the verification to locations within minelev,maxelev.")
   print format_argument("-d dates", "A vector of dates in YYYYMMDD format, e.g.  20130101:20130201.")
   print format_argument("-t times", "A vector of unix timestamps.")
   print format_argument("-fcst", "Which field should be used as the forecast?")
   print format_argument("-l locations", "Limit the verification to these location IDs.")
   print format_argument("-latrange range", "Limit the verification to locations within minlat,maxlat.")
   print format_argument("-lonrange range", "Limit the verification to locations within minlon,maxlon.")
   print format_argument("-o offsets", "Limit the verification to these offsets (in hours).")
   print format_argument("-obs", "Which field should be used as the observation?")
   print format_argument("-r thresholds", "Compute scores for these thresholds (only used by some metrics).")
   print format_argument("-q quantiles", "Compute scores for these quantiles (only used by some metrics).")
   print format_argument("-x dim", "Plot this dimension on the x-axis: date, offset, year, month, location, locationId, elev, lat, lon, threshold, or none. Not supported by all metrics. If not specified, then a default is used based on the metric. 'none' collapses all dimensions and computes one value.")

   # Data manipulation
   print verif.util.green("  Data manipulation:")
   print format_argument("-acc", "Plot accumulated values. Only works for non-derived metrics")
   print format_argument("-agg type", get_aggregation_string())
   print format_argument("-b type", "One of 'below' (< x), 'below=' (<= x), '=within' (<= x < ), 'within' (< x <), 'within=' (< x <=), '=within=' (<= x <=), 'above' (> x), or 'above=' (>= x). For threshold plots (ets, hit, within, etc) 'below/above' computes frequency below/above the threshold, and 'within' computes the frequency between consecutive thresholds.")
   print format_argument("-c file", "File containing climatology data. Subtract all forecasts and obs with climatology values.")
   print format_argument("-C file", "File containing climatology data. Divide all forecasts and obs by climatology values.")
   print format_argument("-hist", "Plot values as histogram. Only works for non-derived metrics")
   print format_argument("-sort", "Plot values sorted. Only works for non-derived metrics")

   # Plot options
   print verif.util.green("  Plotting options:")
   print format_argument("-bottom value", "Bottom boundary location for saved figure [range 0-1]")
   print format_argument("-clim limits", "Force colorbar limits to the two values lower,upper")
   print format_argument("-cmap colormap", "Use this colormap when possible (e.g. jet, inferno, RdBu)")
   print format_argument("-dpi value", "Resolution of image in dots per inch (default 100)")
   print format_argument("-f file", "Save image to this filename")
   print format_argument("-fs size", "Set figure size width,height (in inches). Default 8x6.")
   print format_argument("-labfs size", "Font size for axis labels")
   print format_argument("-lc colors", "Comma-separated list of line colors, such as red,[0.3,0,0],0.3")
   print format_argument("-left value", "Left boundary location for saved figure [range 0-1]")
   print format_argument("-leg titles", "Comma-separated list of legend titles. Use '_' to represent space.")
   print format_argument("-legfs size", "Font size for legend. Set to 0 to hide legend.")
   print format_argument("-legloc loc", "Where should the legend be placed?  Locations such as 'best', 'upper_left', 'lower_right', 'center'. Use underscore when using two words.")
   print format_argument("-lw width", "How wide should lines be?")
   print format_argument("-logx", "Use a logarithmic x-axis")
   print format_argument("-logy", "Use a logarithmic y-axis")
   print format_argument("-maptype", "One of 'simple', 'sat', 'topo', or any of these http://server.arcgisonline.com/arcgis/rest/services names.  'simple' shows a basic ocean/lakes/land map, 'sat' shows a satellite image, and 'topo' a topographical map. Only relevant when '-type map' has been selected.")
   print format_argument("-ms size", "How big should markers be?")
   print format_argument("-nogrid", "Turn the grid on the plot off")
   print format_argument("-nomargin", "Remove margins (whitespace) in the plot")
   print format_argument("-right value", "Right boundary location for saved figure [range 0-1]")
   print format_argument("-simple", "Make a simpler plot, without extra lines, subplots, etc.")
   print format_argument("-sp", "Show a line indicating the perfect score")
   print format_argument("-tickfs size", "Font size for axis ticks")
   print format_argument("-titlefs size", "Font size for title.")
   print format_argument("-title text", "Custom title to chart top")
   print format_argument("-top value", "Top boundary location for saved figure [range 0-1]")
   print format_argument("-type type", "One of 'plot' (default), 'text', 'csv', 'map', or 'maprank'.")
   print format_argument("-xlabel text", "Custom x-axis label")
   print format_argument("-xlim limits", "Force x-axis limits to the two values lower,upper")
   print format_argument("-xticks ticks", "A vector of values to put ticks on the x-axis")
   print format_argument("-xticklabels labels", "A comma-separated list of labels for the x-axis ticks")
   print format_argument("-xrot value", "Rotation angle for x-axis labels")
   print format_argument("-ylabel text", "Custom y-axis label")
   print format_argument("-ylim limits", "Force y-axis limits to the two values lower,upper")
   print format_argument("-yrot value", "Rotation angle for y-axis labels")
   print format_argument("-yticks ticks", "A vector of values to put ticks on the y-axis")
   print format_argument("-xticklabels labels", "A comma-separated list of labels for the y-axis ticks")
   print ""
   print verif.util.green("Metrics (-m):")
   print "  (For a full description of a metric, run verif -m <metric>)"
   metric_types = [verif.metric_type.Deterministic(),
         verif.metric_type.Threshold(),
         verif.metric_type.Probabilistic(),
         verif.metric_type.Diagram()]
   for metric_type in metric_types:
      metrics = verif.metric.get_all_by_type(metric_type)
      outputs = verif.output.get_all_by_type(metric_type)
      metric_outputs = metrics + outputs
      metric_outputs.sort(key=lambda x: x[0].lower(), reverse=False)
      if len(metric_outputs) > 0:
         print verif.util.green("  %s:" % metric_type.description)
         for m in metric_outputs:
            name = m[0].lower()
            if(m[1].is_valid()):
               desc = m[1].description
               print format_argument(name, desc)
   print ""
   print ""
   print verif.util.green("File formats:")
   print format_argument("text", verif.input.Text.description)
   print format_argument("netcdf", verif.input.Netcdf.description)


def format_argument(arg, description, arg_width=19, total_width=None, indent=2):
   """
   Prints formated description to screen, but adds a column for a short descriptor, like this:
               arg            description more description
                              here more more more more more
   | indent | | arg_width   | | total_width                                 |
   """
   if(total_width is None):
      total_width = get_text_width()
   fmt = "%-" + str(indent) + "s%-" + str(arg_width - indent) + "s"
   curr = fmt % ("", arg)
   if(len(arg) > arg_width - indent - 2):
      output = curr + '\n'
      curr = ""
      for i in range(0, arg_width):
         curr = curr + " "
   else:
      output = ""
   lines = description.split('\n')
   for line_num in range(0, len(lines)):
      line = lines[line_num]
      words = line.split()
      for i in range(0, len(words)):
         word = words[i]
         if len(curr) + len(word) >= total_width:
            output = output + curr + "\n"
            curr = ""
            for i in range(0, arg_width):
               curr = curr + " "
         elif(i != 0):
            curr = curr + " "
         curr = curr + word
      output = output + curr
      if line_num < len(lines)-1:
         output = output + "\n"
      curr = " " * arg_width
   return output


def get_text_width():
   """ How wide should the text be output? """
   # return max(50, min(100, get_screen_width()))
   return 80


if __name__ == '__main__':
       main()
