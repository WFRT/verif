from __future__ import print_function
from builtins import str

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
    locations_x = None
    lat_range = None
    lon_range = None
    elev_range = None
    obs_range = None
    thresholds = None
    quantiles = None
    clim_file = None
    clim_type = "subtract"
    leg = None
    ylabel = None
    xlabel = None
    clabel = None
    title = None
    dates = None
    tods = None
    times = None
    leadtimes = None
    axis = None
    aspect = None
    figsize = None
    dpi = 100
    no_margin = False
    bin_type = None
    simple = None
    marker_sizes = None
    line_widths = None
    line_colors = None
    line_styles = None
    markers = None
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
    aggregator_name = None
    do_hist = False
    do_sort = False
    do_acc = False
    xlim = None
    ylim = None
    clim = None
    annotate = False
    xticks = None
    xticklabels = None
    yticks = None
    yticklabels = None
    version = None
    list_thresholds = False
    list_quantiles = False
    list_locations = False
    list_times = False
    list_dates = False
    map_type = None
    xlog = False
    ylog = False
    cmap = None
    obs_leg = None
    obs_field = verif.field.Obs()
    fcst_field = verif.field.Fcst()

    # Parse config files
    i = 1
    extra = []
    while(i < len(argv)):
        arg = argv[i]
        if arg == "--config":
            if i == len(argv) - 1:
                verif.util.error("Missing filename after --config")
            i = i + 1
            filename = argv[i]
            try:
                fid = open(filename, 'r')
                for line in fid:
                    extra += line.split()
            except:
                if not os.path.isfile(filename):
                    verif.util.error("Could not read %s" % filename)
        i = i + 1

    argv = argv + extra

    # Read command line arguments
    i = 1
    while(i < len(argv)):
        arg = argv[i]
        if arg[0] == '-':
            # Process option
            if arg == "-nomargin":
                no_margin = True
            elif arg == "--version":
                version = True
            elif arg == "--list-thresholds":
                list_thresholds = True
            elif arg == "--list-quantiles":
                list_quantiles = True
            elif arg == "--list-locations":
                list_locations = True
            elif arg == "--list-times":
                list_times = True
            elif arg == "--list-dates":
                list_dates = True
            elif arg == "-sp":
                show_perfect = True
            elif arg == "-hist":
                do_hist = True
            elif arg == "-acc":
                do_acc = True
            elif arg == "-sort":
                do_sort = True
            elif arg == "-simple":
                simple = True
            elif arg == "-xlog":
                xlog = True
            elif arg == "-ylog":
                ylog = True
            elif arg == "-a":
                annotate = True
            elif arg == "-nogrid":
                grid = False
            else:
                if len(argv) <= i + 1:
                    verif.util.error("Missing value after %s" % argv[i])
                arg_next = argv[i + 1]
                if arg == "-f":
                    ofile = arg_next
                elif arg == "-l":
                    locations = verif.util.parse_numbers(arg_next)
                elif arg == "-lx":
                    locations_x = verif.util.parse_numbers(arg_next)
                elif arg == "-latrange":
                    lat_range = verif.util.parse_numbers(arg_next)
                elif arg == "-lonrange":
                    lon_range = verif.util.parse_numbers(arg_next)
                elif arg == "-elevrange":
                    elev_range = verif.util.parse_numbers(arg_next)
                elif arg == "-obsrange":
                    obs_range = verif.util.parse_numbers(arg_next)
                elif arg == "-x":
                    axisname = arg_next
                    axis = verif.axis.get(axisname)
                elif arg == "-o":
                    leadtimes = verif.util.parse_numbers(arg_next)
                elif arg == "-leg":
                    leg = verif.util.parse_label(arg_next)
                elif arg == "-ylabel":
                    ylabel = verif.util.parse_label(arg_next)
                elif arg == "-xlabel":
                    xlabel = verif.util.parse_label(arg_next)
                elif arg == "-clabel":
                    clabel = verif.util.parse_label(arg_next)
                elif arg == "-title":
                    title = verif.util.parse_label(arg_next)
                elif arg == "-b":
                    bin_type = arg_next
                elif arg == "-type":
                    plot_type = arg_next
                elif arg == "-fs":
                    figsize = arg_next
                elif arg == "-dpi":
                    dpi = int(arg_next)
                elif arg == "-d":
                    dates = verif.util.parse_numbers(arg_next, True)
                elif arg == "-tod":
                    tods = [int(tod) for tod in verif.util.parse_numbers(arg_next)]
                elif arg == "-t":
                    times = verif.util.parse_numbers(arg_next)
                elif arg == "-c":
                    clim_file = verif.input.get_input(arg_next)
                    clim_type = "subtract"
                elif arg == "-C":
                    clim_file = verif.input.get_input(arg_next)
                    clim_type = "divide"
                elif arg == "-xlim":
                    xlim = verif.util.parse_numbers(arg_next)
                elif arg == "-ylim":
                    ylim = verif.util.parse_numbers(arg_next)
                elif arg == "-clim":
                    clim = verif.util.parse_numbers(arg_next)
                elif arg == "-xticks":
                    xticks = verif.util.parse_numbers(arg_next)
                elif arg == "-xticklabels":
                    xticklabels = verif.util.parse_label(arg_next).split(',')
                elif arg == "-yticks":
                    yticks = verif.util.parse_numbers(arg_next)
                elif arg == "-yticklabels":
                    yticklabels = verif.util.parse_label(arg_next).split(',')
                elif arg == "-agg":
                    aggregator_name = arg_next
                elif arg == "-aspect":
                    aspect = float(arg_next)
                elif arg == "-r":
                    thresholds = np.array(verif.util.parse_numbers(arg_next))
                elif arg == "-q":
                    quantiles = np.array(verif.util.parse_numbers(arg_next))
                    if np.min(quantiles) < 0 or np.max(quantiles) > 1:
                        verif.util.error("Quantiles must be between 0 and 1 inclusive")
                elif arg == "-ms":
                    marker_sizes = verif.util.parse_ints(arg_next)
                elif arg == "-lw":
                    line_widths = verif.util.parse_numbers(arg_next)
                elif arg == "-lc":
                    line_colors = verif.util.parse_colors(arg_next)
                elif arg == "-ls":
                    line_styles = arg_next.split(',')
                elif arg == "-ma":
                    markers = arg_next.split(',')
                elif arg == "-tickfs":
                    tick_font_size = float(arg_next)
                elif arg == "-labfs":
                    lab_font_size = float(arg_next)
                elif arg == "-legfs":
                    leg_font_size = float(arg_next)
                elif arg == "-legloc":
                    leg_loc = arg_next.replace('_', ' ')
                elif arg == "-xrot":
                    xrot = float(arg_next)
                elif arg == "-yrot":
                    yrot = float(arg_next)
                elif arg == "-bottom":
                    bottom_padding = float(arg_next)
                elif arg == "-top":
                    top_padding = float(arg_next)
                elif arg == "-right":
                    right_padding = float(arg_next)
                elif arg == "-left":
                    left_padding = float(arg_next)
                elif arg == "-pad":
                    Pad = arg_next
                elif arg == "-titlefs":
                    title_font_size = float(arg_next)
                elif arg == "-cmap":
                    cmap = arg_next
                elif arg == "-maptype":
                    map_type = arg_next
                    if map_type not in verif.output.allowedMapTypes:
                        verif.util.error("Unknown maptype '%s'" % map_type)
                elif arg == "-obs":
                    obs_field = verif.field.get(arg_next)
                elif arg == "-fcst":
                    fcst_field = verif.field.get(arg_next)
                elif arg == "-obsleg":
                    obs_leg = arg_next
                elif arg == "-m":
                    metric = arg_next
                elif arg == "--config":
                    pass
                else:
                    verif.util.error("Flag '" + argv[i] + "' not recognized")
                i = i + 1
        else:
            ifiles.append(argv[i])
        i = i + 1

    if version:
        print("Version: " + verif.version.__version__)
        return

    # Deal with legend entries
    if leg is not None:
        leg = leg.split(',')
        for i in range(0, len(leg)):
            leg[i] = leg[i].replace('_', ' ')

    if lat_range is not None and len(lat_range) != 2:
        verif.util.error("-latrange <values> must have exactly 2 values")

    if lon_range is not None and len(lon_range) != 2:
        verif.util.error("-lonrange <values> must have exactly 2 values")

    if elev_range is not None and len(elev_range) != 2:
        verif.util.error("-elevrange <values> must have exactly 2 values")

    if obs_range is not None and len(obs_range) != 2:
        verif.util.error("-obsrange <values> must have exactly 2 values")

    if len(ifiles) > 0:
        inputs = [verif.input.get_input(filename) for filename in ifiles]
        data = verif.data.Data(inputs, clim=clim_file, clim_type=clim_type,
              times=times, dates=dates, tods=tods, leadtimes=leadtimes, locations=locations,
              locations_x=locations_x,
              lat_range=lat_range, lon_range=lon_range, elev_range=elev_range,
              obs_range=obs_range, legend=leg, obs_field=obs_field, fcst_field=fcst_field)
    else:
        data = None

    if list_thresholds or list_quantiles or list_locations or list_times or list_dates:
        if len(ifiles) == 0:
            verif.util.error("Files are required in order to list thresholds, quantiles, or times")
        if list_thresholds:
            print("Thresholds:", end=' ')
            for threshold in data.thresholds:
                print("%g" % threshold, end=' ')
            print("")
        if list_quantiles:
            print("Quantiles:", end=' ')
            for quantile in data.quantiles:
                print("%g" % quantile, end=' ')
            print("")
        if list_locations:
            print("    id     lat     lon    elev")
            for location in data.locations:
                print("%6d %7.2f %7.2f %7.1f" % (location.id, location.lat,
                      location.lon, location.elev))
            print("")
        if list_times:
            for time in data.times:
                print("%d" % time)
            print("")
        if list_dates:
            for time in data.times:
                date = verif.util.unixtime_to_date(time)
                diff = time % 86400
                hour = diff / 3600
                minute = (diff % 3600)/60
                second = diff % 60
                print("%d %02d:%02d:%02d" % (date, hour, minute, second))
            print("")
        return
    elif len(ifiles) == 0 and metric is not None:
        m = verif.metric.get(metric)
        if m is not None:
            print(m.help())
        else:
            m = verif.output.get(metric)
            if m is not None:
                print(m.help())
        return
    elif len(argv) == 1 or len(ifiles) == 0 or metric is None:
        print(show_description(data))
        return

    if figsize is not None:
        figsize = figsize.split(',')
        if len(figsize) != 2:
            print("-fs figsize must be in the form: width,height")
            sys.exit(1)

    m = None

    # Handle special plots
    if metric == "pithist":
        pl = verif.output.PitHist()
    elif metric == "obsfcst":
        pl = verif.output.ObsFcst()
    elif metric == "timeseries":
        pl = verif.output.TimeSeries()
    elif metric == "meteo":
        pl = verif.output.Meteo()
    elif metric == "qq":
        pl = verif.output.QQ()
    elif metric == "autocorr":
        pl = verif.output.Auto("corr")
    elif metric == "autocov":
        pl = verif.output.Auto("cov")
    elif metric == "fss":
        pl = verif.output.Fss()
    elif metric == "cond":
        pl = verif.output.Cond()
    elif metric == "against":
        pl = verif.output.Against()
    elif metric == "scatter":
        pl = verif.output.Scatter()
    elif metric == "change":
        pl = verif.output.Change()
    elif metric == "spreadskill":
        pl = verif.output.SpreadSkill()
    elif metric == "taylor":
        pl = verif.output.Taylor()
    elif metric == "error":
        pl = verif.output.Error()
    elif metric == "freq":
        pl = verif.output.Freq()
    elif metric == "roc":
        pl = verif.output.Roc()
    elif metric == "droc":
        pl = verif.output.DRoc()
    elif metric == "droc0":
        pl = verif.output.DRoc0()
    elif metric == "drocnorm":
        pl = verif.output.DRocNorm()
    elif metric == "reliability":
        pl = verif.output.Reliability()
    elif metric == "discrimination":
        pl = verif.output.Discrimination()
    elif metric == "performance":
        pl = verif.output.Performance()
    elif metric == "invreliability":
        pl = verif.output.InvReliability()
    elif metric == "igncontrib":
        pl = verif.output.IgnContrib()
    elif metric == "economicvalue":
        pl = verif.output.EconomicValue()
    elif metric == "marginal":
        pl = verif.output.Marginal()
    else:
        # Standard plots
        # Attempt at automating
        m = verif.metric.get(metric)
        if m is None:
            m = verif.metric.FromField(verif.field.Other(metric))

        if aggregator_name is not None:
            if not m.supports_aggregator:
                verif.util.warning("-m %s does not support -agg" % metric)
            m.aggregator = verif.aggregator.get(aggregator_name)

        # Output type
        if plot_type in ["plot", "text", "csv", "map", "maprank", "rank", "impact", "mapimpact"]:
            if do_sort:
                field = verif.field.get(metric)
                pl = verif.output.Sort(field)
            elif do_hist:
                field = verif.field.get(metric)
                pl = verif.output.Hist(field)
            else:
                pl = verif.output.Standard(m)
        else:
            verif.util.error("Type not understood")

    # Rest dimension of '-x' is not allowed
    if axis is not None and not pl.supports_x:
        verif.util.warning("'-m %s'" % metric + " does not support '-x'. Ignoring it.")
        axis = None

    # Reset dimension if 'threshold' is not allowed
    if axis == verif.axis.Threshold() and ((not pl.supports_threshold) or (m is not None and not m.supports_threshold)):
        verif.util.warning("'-m %s'" % metric + " does not support '-x threshold'. Ignoring it.")
        thresholds = None
        axis = None
    # Reset dimension if 'obs' or 'fcst' is not allowed
    if axis in [verif.axis.Obs(), verif.axis.Fcst()] and ((not pl.supports_field) or (m is not None and not m.supports_field)):
        verif.util.warning("'-m %s'" % metric + " does not support '-x %s'. Ignoring it." % axis.name().lower())
        thresholds = None
        axis = None

    # Create thresholds if needed
    if thresholds is None:
        ttype = None
        if plot_type == "impact":
            ttype = "deterministic"
        elif pl.require_threshold_type == "deterministic":
            ttype = "deterministic"
        elif pl.require_threshold_type == "threshold":
            ttype = "threshold"
        elif m is not None:
            if m.require_threshold_type == "deterministic":
                ttype = "deterministic"
            elif m.require_threshold_type == "threshold":
                ttype = "threshold"
        elif pl.require_threshold_type is not None:
            verif.util.error("Internal error for output %s: Cannot understand required threshold type '%s'" % (pl.name(), pl.require_threshold_type))

        if ttype is not None:
            if ttype == "deterministic":
                smin = np.inf
                smax = -np.inf
                if verif.field.Obs() in data.get_fields():
                    obs = data.get_scores(verif.field.Obs(), 0)
                    smin = min(np.nanmin(obs), smin)
                    smax = max(np.nanmax(obs), smax)
                if verif.field.Fcst() in data.get_fields():
                    fcst = data.get_scores(verif.field.Fcst(), 0)
                    smin = min(np.nanmin(fcst), smin)
                    smax = max(np.nanmax(fcst), smax)
                num_default_thresholds = 20
                thresholds = np.linspace(smin, smax, num_default_thresholds)
                verif.util.warning("Missing '-r <thresholds>'. Automatically setting thresholds: %s" % (','.join(["%g" % q for q in thresholds])))
            elif ttype == "threshold":
                thresholds = data.thresholds
                verif.util.warning("Missing '-r <thresholds>'. Automatically setting thresholds: %s" % (','.join(["%g" % q for q in thresholds])))
            if len(thresholds) == 0:
                verif.util.error("No thresholds available")
    ttype = None
    if pl.require_threshold_type == "quantile":
        ttype = "quantile"
    elif m is not None:
        if m.require_threshold_type == "quantile":
            ttype = "quantile"

    if ttype == "quantile":
        if quantiles is None:
            quantiles = data.quantiles
            verif.util.warning("Missing '-q <quantiles>'. Automatically setting quantiles: %s" % (','.join(["%g" % q for q in quantiles])))
        # TODO: This is a bit of a hack, using thresholds to hold the
        # quantiles. But otherwise, the classes in output need to deal with
        # testing if the metric needs thresholds or quantiles
        thresholds = quantiles

    # Set plot parameters
    if simple is not None:
        pl.simple = simple
    if marker_sizes is not None:
        pl.ms = marker_sizes
    if line_widths is not None:
        pl.lw = line_widths
    if line_colors is not None:
        pl.line_colors = line_colors
    if line_styles is not None:
        pl.line_styles = line_styles
    if markers is not None:
        pl.markers = markers
    if lab_font_size is not None:
        pl.labfs = lab_font_size
    if leg_font_size is not None:
        pl.legfs = leg_font_size
    if title_font_size is not None:
        pl.title_font_size = title_font_size
    if leg_loc is not None:
        pl.leg_loc = leg_loc
    if tick_font_size is not None:
        pl.tick_font_size = tick_font_size
    if xrot is not None:
        pl.xrot = xrot
    if yrot is not None:
        pl.yrot = yrot
    if bottom_padding is not None:
        pl.bottom = bottom_padding
    if top_padding is not None:
        pl.top = top_padding
    if right_padding is not None:
        pl.right = right_padding
    if left_padding is not None:
        pl.left = left_padding
    if Pad is not None:
        pl.pad = None
    if bin_type is not None:
        pl.bin_type = bin_type
    if show_perfect is not None:
        pl.show_perfect = show_perfect
    if xlim is not None:
        pl.xlim = xlim
    if ylim is not None:
        pl.ylim = ylim
    if clim is not None:
        pl.clim = clim
    if xticks is not None:
        pl.xticks = xticks
    if xticklabels is not None:
        pl.xticklabels = xticklabels
    if yticks is not None:
        pl.yticks = yticks
    if yticklabels is not None:
        pl.yticklabels = yticklabels
    if xlog is not None:
        pl.xlog = xlog
    if ylog is not None:
        pl.ylog = ylog
    if annotate is not None:
        pl.annotate = annotate
    pl.grid = grid
    if cmap is not None:
        pl.cmap = cmap
    if map_type is not None:
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
    if aggregator_name is not None:
        pl.aggregator = verif.aggregator.get(aggregator_name)
    if aspect is not None:
        pl.aspect = aspect
    pl.show_margin = not no_margin
    if ylabel is not None:
        pl.ylabel = ylabel
    if xlabel is not None:
        pl.xlabel = xlabel
    if clabel is not None:
        pl.clabel = clabel
    if title is not None:
        pl.title = title
    if obs_leg is not None:
        pl.obs_leg = obs_leg
    if do_acc:
        if pl.supports_acc:
            pl.show_acc = do_acc
        else:
            verif.util.warning("%s does not support -acc" % metric)

    if plot_type == "text":
        pl.text(data)
    elif plot_type == "csv":
        pl.csv(data)
    elif plot_type == "map":
        pl.map(data)
    elif plot_type == "maprank":
        pl.show_rank = True
        pl.map(data)
    elif plot_type == "rank":
        pl.show_rank = True
        pl.plot_rank(data)
    elif plot_type == "impact":
        pl.plot_impact(data)
    elif plot_type == "mapimpact":
        pl.plot_mapimpact(data)
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
    desc = "Program to compute verification scores for weather forecasts. See https://github.com/WFRT/verif/wiki for how to format input files."
    s = "usage: verif files -m metric [options]\n"
    s += "\n"
    s += textwrap.fill(desc, get_text_width()) + "\n"
    s += "\n"
    s += verif.util.green("Arguments:") + "\n"
    s += format_argument("files", "One or more verification files in NetCDF or text format (see 'File Formats' below). The file format is autodetected.") + "\n"
    s += format_argument("-m metric", "Which verification metric to use? See 'Metrics' below.") + "\n"
    s += format_argument("--config file", "Read further arguments from this file. This flag can appear multiple times.") + "\n"
    s += format_argument("--list-times", "Prints what times are available in the files") + "\n"
    s += format_argument("--list-dates", "Like --list-times but in YYYYMMDD HH:MM:SS format") + "\n"
    s += format_argument("--list-locations", "Prints what locations are available in the files") + "\n"
    s += format_argument("--list-quantiles", "Prints what quantiles are available in the files") + "\n"
    s += format_argument("--list-thresholds", "Prints what thresholds are available in the files") + "\n"
    s += format_argument("--version", "Prints what version of verif this is") + "\n"
    # Dimensions
    s += verif.util.green("  Dimensions and subset:") + "\n"
    s += "  (Note: vectors can be entered using commas, or MATLAB syntax i.e 3:5 is 3,4,5 and 3:2:7 is 3,5,7)\n"
    s += format_argument("-d dates", "A vector of dates in YYYYMMDD format, e.g.  20130101:20130201.") + "\n"
    s += format_argument("-elevrange range", "Limit the verification to locations within minelev,maxelev.") + "\n"
    s += format_argument("-l locations", "Limit the verification to these location IDs.") + "\n"
    s += format_argument("-lx locations", "Remove these locations from the verification. This happens after -l, -latrange, -lonrange, and -elevrange has been applied.") + "\n"
    s += format_argument("-latrange range", "Limit the verification to locations within minlat,maxlat.") + "\n"
    s += format_argument("-lonrange range", "Limit the verification to locations within minlon,maxlon.") + "\n"
    s += format_argument("-o leadtimes", "Limit the verification to these leadtimes (in hours).") + "\n"
    s += format_argument("-obsrange range", "Limit the verification to this range of observation values.") + "\n"
    s += format_argument("-r thresholds", "Compute scores using these thresholds (only used by some metrics).") + "\n"
    s += format_argument("-q quantiles", "Compute scores using these quantiles (only used by some metrics).") + "\n"
    s += format_argument("-t times", "A vector of unix timestamps. Only allow these times.") + "\n"
    s += format_argument("-tod timeofdays", "A vector of hours of day, e.g.  0:12. Only allow times with these hour of the day.") + "\n"
    s += format_argument("-x dim", "Plot this dimension on the x-axis: time, leadtime, year, month, week, day, timeofday, dayofyear, monthofyear, location, elev, lat, lon, threshold, leadtimeday, or no. Not supported by all metrics. If not specified, then a default is used based on the metric. 'location' refers to the location id. 'leadtimeday' aggregates leadtimes into whole forecast days. 'no' collapses all dimensions and computes one value.") + "\n"

    # Data manipulation
    s += verif.util.green("  Data manipulation:") + "\n"
    s += format_argument("-acc", "Accumulated values along the x-axis. Does not work for 'Special diagams.'") + "\n"
    s += format_argument("-agg type", get_aggregation_string()) + "\n"
    s += format_argument("-b type", "One of 'below' (< x), 'below=' (<= x), '=within' (<= x < ), 'within' (< x <), 'within=' (< x <=), '=within=' (<= x <=), 'above' (> x), or 'above=' (>= x). For threshold plots (ets, hit, within, etc) 'below/above' computes frequency below/above the threshold, and 'within' computes the frequency between consecutive thresholds.") + "\n"
    s += format_argument("-c file", "File containing climatology data. Subtract all forecasts and obs with climatology values.") + "\n"
    s += format_argument("-C file", "File containing climatology data. Divide all forecasts and obs by climatology values.") + "\n"
    s += format_argument("-fcst field", "What variable should be used as the forecast? 'obs', 'fcst' (default), threshold:<threshold>, quantile:<quantile>, 'pit', or the name of any other field in the input files.") + "\n"
    s += format_argument("-hist", "Plot values as histogram. Only works for any field that can be specified with -fcst.") + "\n"
    s += format_argument("-obs field", "What variable should be used as the observation? See -fcst.") + "\n"
    s += format_argument("-sort", "Plot values sorted. Only works for any field than can be specified with -fcst.") + "\n"

    # Plot options
    s += verif.util.green("  Plotting options:") + "\n"
    s += format_argument("-a", "Annotate graph by labeling each data point. Not supported by all metrics.") + "\n"
    s += format_argument("-aspect ratio", "Force the aspect ratio of the plot. A value greater than 1 will stretch out the y-axis.") + "\n"
    s += format_argument("-bottom value", "Bottom boundary location for saved figure [range 0-1]") + "\n"
    s += format_argument("-clabel text", "Custom colorbar label") + "\n"
    s += format_argument("-clim limits", "Force colorbar limits to the two values lower,upper. Only used in combination with -type map.") + "\n"
    s += format_argument("-cmap colormap", "Use this colormap when possible (e.g. jet, inferno, RdBu). Only used in combination with -type map.") + "\n"
    s += format_argument("-dpi value", "Resolution of image in dots per inch (default 100)") + "\n"
    s += format_argument("-f file", "Save image to this filename") + "\n"
    s += format_argument("-fs size", "Set figure size width,height (in inches). Default 8x6.") + "\n"
    s += format_argument("-labfs size", "Font size for axis labels") + "\n"
    s += format_argument("-lc colors", "Comma-separated list of line colors, such as red,[0.3,0,0],0.3. Colors are repeated if there are more lines than colors.") + "\n"
    s += format_argument("-left value", "Left boundary location for saved figure [range 0-1]") + "\n"
    s += format_argument("-leg titles", "Comma-separated list of legend titles. Use '_' to represent space.") + "\n"
    s += format_argument("-legfs size", "Font size for legend. Set to 0 to hide legend.") + "\n"
    s += format_argument("-legloc loc", "Where should the legend be placed?  Locations such as 'best', 'upper_left', 'lower_right', 'center'. Use underscore when using two words.") + "\n"
    s += format_argument("-ls styles", "Comma-separated list of line styles, such as -,--. Styles are repeated if there are more lines than styles.") + "\n"
    s += format_argument("-lw width", "Comma-separated list of line widths") + "\n"
    s += format_argument("-maptype type", "One of 'simple', 'sat', or 'topo'.  'simple' shows a basic ocean/lakes/land map, 'sat' shows a satellite image, and 'topo' a topographical map. Only relevant when '-type map' has been selected.") + "\n"
    s += format_argument("-ma markers", "Comma-separated list of markers (e.g.  o,*,x)") + "\n"
    s += format_argument("-ms size", "Comma-separated list of marker sizes") + "\n"
    s += format_argument("-nogrid", "Turn the grid on the plot off") + "\n"
    s += format_argument("-nomargin", "Remove margins (whitespace) in the plot") + "\n"
    s += format_argument("-obsleg", "Name to put in legend for observations (if applicable)") + "\n"
    s += format_argument("-right value", "Right boundary location for saved figure [range 0-1]. Must be greater than -left.") + "\n"
    s += format_argument("-simple", "Make a simpler plot, without extra lines, subplots, etc.") + "\n"
    s += format_argument("-sp", "Show a line indicating the perfect score") + "\n"
    s += format_argument("-tickfs size", "Font size for axis ticks") + "\n"
    s += format_argument("-title text", "Custom title to chart top") + "\n"
    s += format_argument("-titlefs size", "Font size for title.") + "\n"
    s += format_argument("-top value", "Top boundary location for saved figure [range 0-1].  Must be greater than -bottom.") + "\n"
    s += format_argument("-type type", "One of 'plot' (default), 'text', 'csv', 'map', 'rank', 'maprank', 'impact', or 'mapimpact'.") + "\n"
    s += format_argument("-xlabel text", "Custom x-axis label") + "\n"
    s += format_argument("-xlim limits", "Force x-axis limits to the two values lower,upper") + "\n"
    s += format_argument("-xlog", "Use a logarithmic x-axis") + "\n"
    s += format_argument("-xrot value", "Rotation angle for x-axis labels") + "\n"
    s += format_argument("-xticks ticks", "A vector of values to put ticks on the x-axis") + "\n"
    s += format_argument("-xticklabels labs", "A comma-separated list of labels for the x-axis ticks") + "\n"
    s += format_argument("-ylabel text", "Custom y-axis label") + "\n"
    s += format_argument("-ylim limits", "Force y-axis limits to the two values lower,upper") + "\n"
    s += format_argument("-ylog", "Use a logarithmic y-axis") + "\n"
    s += format_argument("-yrot value", "Rotation angle for y-axis labels") + "\n"
    s += format_argument("-yticks ticks", "A vector of values to put ticks on the y-axis") + "\n"
    s += format_argument("-yticklabels labs", "A comma-separated list of labels for the y-axis ticks") + "\n"
    s += "\n"
    s += verif.util.green("Metrics (-m):") + "\n"
    s += "  (For a full description of a metric, run verif -m <metric>)\n"
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
            s += verif.util.green("  %s:" % metric_type.description) + "\n"
            for m in metric_outputs:
                name = m[0].lower()
                if m[1].is_valid():
                    desc = m[1].description
                    s += format_argument(name, desc) + "\n"

    return s


def format_argument(arg, description, arg_width=22, total_width=None, indent=2):
    """
    Prints formated description to screen, but adds a column for a short descriptor, like this:
                arg            description more description
                               here more more more more more
    | indent | | arg_width   | | total_width                                 |
    """
    if total_width is None:
        total_width = get_text_width()
    fmt = "%-" + str(indent) + "s%-" + str(arg_width - indent) + "s"
    curr = fmt % ("", arg)
    if len(arg) > arg_width - indent - 2:
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
            elif i != 0:
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
    run(sys.argv)
