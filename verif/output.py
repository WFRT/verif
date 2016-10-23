# -*- coding: ISO-8859-1 -*-
import matplotlib.pyplot as mpl
import re
import datetime
import verif.util
import verif.metric
import numpy as np
import os
import inspect
import sys
import matplotlib.dates as mpldates
reload(sys)
sys.setdefaultencoding('ISO-8859-1')

allowedMapTypes = ["simple", "sat", "topo", "ESRI_Imagery_World_2D",
         "ESRI_StreetMap_World_2D", "I3_Imagery_Prime_World",
         "NASA_CloudCover_World", "NatGeo_World_Map", "NGS_Topo_US_2D",
         "Ocean_Basemap", "USA_Topo_Maps", "World_Imagery",
         "World_Physical_Map", "World_Shaded_Relief", "World_Street_Map",
         "World_Terrain_Base", "World_Topo_Map"]


def get_all():
   """
   Returns a dictionary of all output classes where the key is the class
   name (string) and the value is the class object
   """
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return temp


def get_output(name):
   """ Returns an instance of an object with the given class name """
   outputs = get_all()
   o = None
   for output in outputs:
      if(name == output[0].lower() and output[1].is_valid()):
         o = output[1]()
   return o


class Output(object):
   """
   Abstract class representing a plot

   usage:
   output = verif.output.Qq()
   output.plot(data)
   """
   _description = ""
   _default_axis = "offset"
   _default_bin_type = "above"
   _reqThreshold = False
   _supThreshold = True
   _supX = True
   _experimental = False
   _legLoc = "best"  # Where should the legend go?
   _logX = False
   _logY = False
   _reference = None
   _long = None

   def __init__(self):
      self._filename = None
      self._thresholds = [None]
      leg = None
      self.default_lines = ['-', '-', '-', '--']
      self.default_markers = ['o', '', '.', '']
      self.default_colors = ['r', 'b', 'g', [1, 0.73, 0.2], 'k']
      self._lc = None
      self._ls = None
      self.colors = None
      self.styles = None
      self._ms = 8
      self._lw = 2
      self._labfs = 16
      self._tickfs = 16
      self._legfs = 16
      self._titlefs = 16
      self._figsize = [5, 8]
      self._showMargin = True
      self._xrot = 0
      self._minlth = None
      self._majlth = None
      self._majwid = None
      self._bot = None
      self._top = None
      self._left = None
      self._right = None
      self._xaxis = self.default_axis()
      self._binType = self.default_bin_type()
      self._showPerfect = False
      self._dpi = 100
      self._xlim = None
      self._ylim = None
      self._clim = None
      self._xticks = None
      self._yticks = None
      self._title = None
      self._xlabel = None
      self._ylabel = None
      self._tight = False
      self._simple = False
      self._mapType = None
      self._cmap = mpl.cm.jet

   def set_axis(self, axis):
      """ Produce output independently for each value along this axis """
      if(axis is not None):
         self._xaxis = axis

   def set_bin_type(self, binType):
      if(binType is not None):
         self._binType = binType

   def set_thresholds(self, thresholds):
      if(thresholds is None):
         thresholds = [None]
      thresholds = np.array(thresholds)
      self._thresholds = thresholds

   def set_fig_size(self, size):
      self._figsize = size

   def set_filename(self, filename):
      """
      When set, output the figure to this filename. File extension is
      auto-detected.
      """
      self._filename = filename

   def set_dpi(self, dpi):
      """ Sets the number of dots per inch if output to file """
      self._dpi = dpi

   def set_leg_loc(self, legLoc):
      self._legLoc = legLoc

   def set_show_margin(self, showMargin):
      self._showMargin = showMargin

   def xlim(self, lim=None):
      """
      Set/get the x-axis limits for the output

      lim      A two-element list, with a lower and upper value
      """
      if lim is None:
         return self._xlim
      if(len(lim) != 2):
         verif.util.error("xlim must be a vector of length 2")
      self._xlim = lim

   def ylim(self, lim=None):
      """
      Set/get the x-axis limits for the output

      lim      A two-element list, with a lower and upper value
      """
      if lim is None:
         return self._ylim
      if(len(lim) != 2):
         verif.util.error("ylim must be a vector of length 2")
      self._ylim = lim

   def clim(self, lim):
      """
      Set/get the range of values that any colormap should use

      lim      A two-element list, with a lower and upper value
      """
      if lim is None:
         return self._clim
      if(len(lim) != 2):
         verif.util.error("clim must be a vector of length 2")
      self._clim = lim

   def xticks(self, ticks):
      """
      Set/get the x-axis values where ticks will be placed

      ticks      A list or numpy array of ticks
      """
      self._xticks = ticks

   def yticks(self, ticks):
      """
      Set/get the y-axis values where ticks will be placed

      ticks      A list or numpy array of ticks
      """
      self._yticks = ticks

   def set_simple(self, flag):
      self._simple = flag

   def ms(self, ms):
      """ Set/get the size of any markers used """
      self._ms = ms

   def lw(self, lw):
      """ Set/get the width of any lines used """
      self._lw = lw

   def ylabel(self, ylabel):
      """ Set/get the y-axis label """
      self._ylabel = ylabel

   def xlabel(self, xlabel):
      """ Set/get the x-axis label """
      self._xlabel = xlabel

   def title(self, title):
      """ Set/get the title of the figure """
      self._title = title

   def set_line_colors(self, lc):
      self._lc = lc

   def set_line_style(self, ls):
      self._ls = ls

   def set_tick_font_size(self, fs):
      self._tickfs = fs

   def set_lab_font_size(self, fs):
      self._labfs = fs

   def set_leg_font_size(self, fs):
      self._legfs = fs

   def set_title_font_size(self, fs):
      self._titlefs = fs

   def set_x_rotation(self, xrot):
      self._xrot = xrot

   def set_minor_length(self, minlth):
      self._minlth = minlth

   def set_major_length(self, majlth):
      self._majlth = majlth

   def set_major_width(self, majwid):
      self._majwid = majwid

   def set_bottom(self, bot):
      self._bot = bot

   def set_top(self, top):
      self._top = top

   def set_left(self, left):
      self._left = left

   def set_right(self, right):
      self._right = right

   # def set_pad(self, pad):
   #   self._pad = pad

   def set_show_perfect(self, showPerfect):
      self._showPerfect = showPerfect

   def set_tight(self, tight):
      self._tight = tight

   def set_aggregator_name(self, name):
      self._aggregatorName = name

   def set_map_type(self, type):
      if type not in allowedMapTypes:
         verif.util.error("Map type '%s' not recognized. Must be one of %s" % (type,
            allowedMapTypes))
      self._mapType = type

   def set_log_x(self, flag):
      self._logX = flag

   def set_log_y(self, flag):
      self._logY = flag

   def set_cmap(self, cmap):
      if isinstance(cmap, basestring):
         cmap = mpl.cm.get_cmap(cmap)
      self._cmap = cmap

   @classmethod
   def default_axis(cls):
      return cls._default_axis

   @classmethod
   def default_bin_type(cls):
      return cls._default_bin_type

   @classmethod
   def requires_thresholds(cls):
      return cls._reqThreshold

   @classmethod
   def supports_x(cls):
      return cls._supX

   @classmethod
   def supports_threshold(cls):
      return cls._supThreshold

   @classmethod
   def get_class_name(cls):
      name = cls.__name__
      return name

   @classmethod
   def description(cls):
      extra = ""
      # if(cls._experimental):
      #    extra = " " + verif.util.experimental()
      return cls._description + extra

   # Is this a valid output that should be created be called?
   @classmethod
   def is_valid(cls):
      return cls.summary() is not ""

   @classmethod
   def reference(cls):
      return cls._reference

   @classmethod
   def help(cls):
      s = cls.description()
      if(cls._long is not None):
         s = s + "\n" + verif.util.green("Description: ") + cls._long
      if(cls.reference() is not None):
         s = s + "\n" + verif.util.green("Reference: ") + cls.reference()
      return s

   @classmethod
   def summary(cls):
      return cls.description()

   # Public
   # Call this to create a plot, saves to file
   def plot(self, data):
      self._plot_core(data)
      self._adjust_axes(data)
      self._legend(data)
      self._save_plot(data)

   # Call this to write text output
   def text(self, data):
      self._text_core(data)

   # Call this to write csv output
   def csv(self, data):
      self._csv_core(data)

   # Draws a map of the data
   def map(self, data):
      self._map_core(data)
      # self._legend(data)
      self._save_plot(data)

   def _plot_perfect_score(self, x, perfect, color="gray", zorder=-1000):
      if(perfect is None):
         return
      if(self._showPerfect):
         # Make 'perfect' same length as 'x'
         if(not hasattr(perfect, "__len__")):
            perfect = perfect * np.ones(len(x), 'float')
         mpl.plot(x, perfect, '-', lw=7, color=color, label="ideal",
               zorder=zorder)

   # Implement these methods
   def _plot_core(self, data):
      verif.util.error("This type does not plot")

   def _text_core(self, data):
      verif.util.error("This type does not output text")

   def _csv_core(self, data):
      verif.util.error("This type does not output csv")

   def _map_core(self, data):
      verif.util.error("This type does not produce maps")

   # Helper functions
   def _get_color(self, i, total):
      if(self._lc is not None):
         firstList = self._lc.split(",")
         numList = []
         finalList = []

         for string in firstList:
            if("[" in string):   # for rgba args
               if(not numList):
                  string = string.replace("[", "")
                  numList.append(float(string))
               else:
                  verif.util.error("Invalid rgba arg \"{}\"".format(string))

            elif("]" in string):
               if(numList):
                  string = string.replace("]", "")
                  numList.append(float(string))
                  finalList.append(numList)
                  numList = []
               else:
                  verif.util.error("Invalid rgba arg \"{}\"".format(string))

            # append to rgba lists if present, otherwise grayscale intensity
            elif(verif.util.is_number(string)):
               if(numList):
                  numList.append(float(string))
               else:
                  finalList.append(string)

            else:
               if(not numList):  # string args and hexcodes
                  finalList.append(string)
               else:
                  verif.util.error("Cannot read color args.")
         self.colors = finalList
         return self.colors[i % len(self.colors)]

      # use default colours if no colour input given
      else:
         self.colors = self.default_colors
         return self.colors[i % len(self.default_colors)]

   def _get_style(self, i, total, connectingLine=True, lineOnly=False):
      if(self._ls is not None):
         listStyles = self._ls.split(",")
         # loop through input linestyles (independent of colors)
         I = i % len(listStyles)
         return listStyles[I]

      else:  # default linestyles
         I = (i / len(self.colors)) % len(self.default_lines)
         line = self.default_lines[I]
         marker = self.default_markers[I]
         if(lineOnly):
            return line
         if(connectingLine):
            return line + marker
         return marker

   # Saves to file, set figure size
   def _save_plot(self, data):
      if(self._figsize is not None):
         mpl.gcf().set_size_inches(int(self._figsize[0]),
                                   int(self._figsize[1]))
      if(not self._showMargin):
         verif.util.remove_margin()
      if(self._filename is not None):
         mpl.savefig(self._filename, bbox_inches='tight', dpi=self._dpi)
      else:
         fig = mpl.gcf()
         fig.canvas.set_window_title(data.get_filenames()[0])
         mpl.show()

   def _legend(self, data, names=None):
      if(self._legfs > 0):
         if(names is None):
            mpl.legend(loc=self._legLoc, prop={'size': self._legfs})
         else:
            mpl.legend(names, loc=self._legLoc, prop={'size': self._legfs})

   def _get_threshold_limits(self, thresholds):
      x = thresholds
      if(self._binType == "below"):
         lowerT = [-np.inf for i in range(0, len(thresholds))]
         upperT = thresholds
      elif(self._binType == "above"):
         lowerT = thresholds
         upperT = [np.inf for i in range(0, len(thresholds))]
      elif(self._binType == "within"):
         lowerT = thresholds[0:-1]
         upperT = thresholds[1:]
         x = [(lowerT[i] + upperT[i]) / 2 for i in range(0, len(lowerT))]
      else:
         verif.util.error("Unrecognized bintype")
      return [lowerT, upperT, x]

   def _set_y_axis_limits(self, metric):
      currYlim = mpl.ylim()
      ylim = [metric.min(), metric.max()]
      if(ylim[0] is None):
         ylim[0] = currYlim[0]
      if(ylim[1] is None):
         ylim[1] = currYlim[1]
      mpl.ylim(ylim)

   def _adjust_axes(self, data):
      # Apply adjustements to all subplots
      for ax in mpl.gcf().get_axes():
         ax.set_title(ax.get_title(), fontsize=self._titlefs)
         # Tick font sizes
         for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(self._tickfs)
         for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(self._tickfs)
         ax.set_xlabel(ax.get_xlabel(), fontsize=self._labfs)
         ax.set_ylabel(ax.get_ylabel(), fontsize=self._labfs)
         # mpl.rcParams['axes.labelsize'] = self._labfs

         # Tick lines
         if(self._minlth is not None):
            mpl.tick_params('both', length=self._minlth, which='minor')
         if(self._majlth is not None):
            mpl.tick_params('both', length=self._majlth, width=self._majwid,
                  which='major')
         for label in ax.get_xticklabels():
            label.set_rotation(self._xrot)

      for ax in mpl.gcf().get_axes():
         if(self._xlim is not None):
            xlim = self._xlim
            # Convert date to datetime objects
            if(data.is_axis_date()):
               xlim = verif.util.convert_dates(xlim)
            mpl.xlim(xlim)
         if(self._ylim is not None):
            mpl.ylim(self._ylim)
         if(self._clim is not None):
            mpl.clim(self._clim)
         if self._logX:
            ax.set_xscale('log')
         if self._logY:
            ax.set_yscale('log')

      # Labels
      if(self._xlabel is not None):
         mpl.xlabel(self._xlabel)
      if(self._ylabel is not None):
         mpl.ylabel(self._ylabel)
      if(self._title is not None):
         mpl.title(self._title, fontsize=self._titlefs)

      # Ticks
      if(self._xticks is not None):
         mpl.xticks(self._xticks)
      if(self._yticks is not None):
         mpl.yticks(self._yticks)

      # Margins
      mpl.gcf().subplots_adjust(bottom=self._bot, top=self._top,
            left=self._left, right=self._right)

   def _plot_obs(self, x, y, isCont=True, zorder=0, label="obs"):
      if(isCont):
         mpl.plot(x, y, ".-", color="gray", lw=5, label=label, zorder=zorder)
      else:
         mpl.plot(x, y, "o", color="gray", ms=self._ms, label=label,
               zorder=zorder)

   def _draw_circle(self, radius, xcenter=0, ycenter=0, maxradius=np.inf,
         style="--", color="k", lw=1, label="", zorder=-100):
      """
      Draws a circle
      radius      Radius of circle
      xcentre     x-axis value of centre of circle
      ycentre     y-axis value of centre of circle
      maxradius   Don't let the circle go outside an envelope circle with this
                  radius (centered on the origin)
      label       Use this text in the legend
      """
      angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
      x = np.sin(angles) * radius + xcenter
      y = np.cos(angles) * radius + ycenter

      # Only keep points within the circle
      I = np.where(x ** 2 + y ** 2 < maxradius ** 2)[0]
      if(len(I) == 0):
         return
      x = x[I]
      y = y[I]
      mpl.plot(x, y, style, color=color, lw=lw, zorder=zorder, label=label)
      mpl.plot(x, -y, style, color=color, lw=lw, zorder=zorder)

   def _plot_confidence(self, x, y, variance, n, color):
      # variance = y*(1-y) # For bins

      # Remove missing points
      I = np.where(n != 0)[0]
      if(len(I) == 0):
         return
      x = x[I]
      y = y[I]
      variance = variance[I]
      n = n[I]

      z = 1.96  # 95% confidence interval
      type = "wilson"
      style = "--"
      if type == "normal":
         mean = y
         lower = mean - z * np.sqrt(variance / n)
         upper = mean + z * np.sqrt(variance / n)
      elif type == "wilson":
         mean = 1 / (1 + 1.0 / n * z ** 2) * (y + 0.5 * z ** 2 / n)
         upper = mean + 1 / (1 + 1.0 / n * z ** 2) * z * np.sqrt(variance / n +
               0.25 * z ** 2 / n ** 2)
         lower = mean - 1 / (1 + 1.0 / n * z ** 2) * z * np.sqrt(variance / n +
               0.25 * z ** 2 / n ** 2)
      mpl.plot(x, upper, style, color=color, lw=self._lw, ms=self._ms,
            label="")
      mpl.plot(x, lower, style, color=color, lw=self._lw, ms=self._ms,
            label="")
      verif.util.fill(x, lower, upper, color, alpha=0.3)


class Default(Output):
   """
   Plot a metric from verif.metric
   """
   _legLoc = "upper left"

   def __init__(self, metric):
      """
      metric      an metric object from verif.metric
      """
      Output.__init__(self)
      self._metric = metric
      if(metric.default_axis() is not None):
         self._xaxis = metric.default_axis()
      if(metric.default_bin_type() is not None):
         self._binType = metric.default_bin_type()
      self._showRank = False
      self._showAcc = False
      self._set_leg_sort = False
      self._showSmoothingLine = False  # Draw a smoothed line through the points

      # Settings
      self._mapLowerPerc = 0    # Lower percentile (%) to show in colourmap
      self._mapUpperPerc = 100  # Upper percentile (%) to show in colourmap
      self._mapLabelLocations = True  # Show locationIds in map?
      self._minLatLonRange = 0.001  # What is the smallest map size allowed (in degrees)

   def set_show_rank(self, showRank):
      self._showRank = showRank

   def set_leg_sort(self, dls):
      self._set_leg_sort = dls

   def set_show_acc(self, showAcc):
      self._showAcc = showAcc

   def get_x_y(self, data):
      thresholds = self._thresholds
      axis = data.get_axis()

      [lowerT, upperT, xx] = self._get_threshold_limits(thresholds)
      if(axis != "threshold"):
         xx = data.get_axis_values()

      filenames = data.get_filenames()
      F = data.get_num_files()
      y = None
      x = None
      for f in range(0, F):
         data.set_file_index(f)
         yy = np.zeros(len(xx), 'float')
         if(axis == "threshold"):
            for i in range(0, len(lowerT)):
               yy[i] = self._metric.compute(data, [lowerT[i], upperT[i]])
         else:
            for i in range(0, len(lowerT)):
               yy = yy + self._metric.compute(data, [lowerT[i], upperT[i]])
            yy = yy / len(thresholds)

         if(sum(np.isnan(yy)) == len(yy)):
            verif.util.warning("No valid scores for " + filenames[f])
         if(y is None):
            y = np.zeros([F, len(yy)], 'float')
            x = np.zeros([F, len(xx)], 'float')
         y[f, :] = yy
         x[f, :] = xx
         if(self._showAcc):
            y[f, :] = np.nan_to_num(y[f, :])
            y[f, :] = np.cumsum(y[f, :])
      return [x, y]

   def _legend(self, data, names=None):
      if(self._legfs > 0):
         mpl.legend(loc=self._legLoc, prop={'size': self._legfs})

   def _plot_core(self, data):

      data.set_axis(self._xaxis)

      # We have to derive the legend list here, because we might want to
      # specify the order
      labels = np.array(data.get_legend())

      F = data.get_num_files()
      [x, y] = self.get_x_y(data)

      # Sort legend entries such that the appear in the same order as the
      # y-values of the lines
      if(self._set_leg_sort):
         if(not self._showAcc):
            # averaging for non-acc plots
            averages = (verif.util.nanmean(y, axis=1))
            ids = averages.argsort()[::-1]

         else:
            ends = y[:, -1]  # take last points for acc plots
            ids = ends.argsort()[::-1]

         labels = [labels[i] for i in ids]

      else:
         ids = range(0, F)

      if(self._xaxis == "none"):
         w = 0.8
         x = np.linspace(1 - w / 2, len(y) - w / 2, len(y))
         mpl.bar(x, y, color='w', lw=self._lw)
         mpl.xticks(range(1, len(y) + 1), labels)
      else:
         for f in range(0, F):
            # colors and styles to follow labels
            color = self._get_color(ids[f], F)
            style = self._get_style(ids[f], F, data.is_axis_continuous())
            alpha = (1 if(data.is_axis_continuous()) else 0.55)
            mpl.plot(x[ids[f]], y[ids[f]], style, color=color,
                  label=labels[f], lw=self._lw, ms=self._ms,
                  alpha=alpha)
            if self._showSmoothingLine:
               from scipy import ndimage
               I = np.argsort(x[ids[f]])
               xx = np.sort(x[ids[f]])
               yy = y[ids[f]][I]
               I = np.where((np.isnan(xx) == 0) & (np.isnan(yy) == 0))[0]
               xx = xx[I]
               yy = yy[I]
               N = 21
               yy = ndimage.convolve(yy, 1.0/N*np.ones(N), mode="mirror")
               mpl.plot(xx, yy, "--", color=color, lw=self._lw, ms=self._ms)

         mpl.xlabel(data.get_axis_label())
         mpl.ylabel(self._metric.label(data))

         if(data.is_axis_date()):
            mpl.gca().xaxis_date()
         else:
            mpl.gca().xaxis.set_major_formatter(data.get_axis_formatter())
            # NOTE: Don't call the locator on a date axis
            mpl.gca().xaxis.set_major_locator(data.get_axis_locator())
         perfect_score = self._metric.perfect_score()
         self._plot_perfect_score(x[0], perfect_score)

      mpl.grid()
      if(not self._showAcc):
         self._set_y_axis_limits(self._metric)

      if(self._tight):
         oldTicks = mpl.gca().get_xticks()
         diff = oldTicks[1] - oldTicks[0]  # keep auto tick interval
         tickRange = np.arange(round(np.min(x)), round(np.max(x)) + diff, diff)
         # make new ticks, to start from the first day of the desired interval
         mpl.gca().set_xticks(tickRange)
         mpl.autoscale(enable=True, axis=u'x', tight=True)  # make xaxis tight

   def _text_core(self, data):
      thresholds = self._thresholds

      data.set_axis(self._xaxis)

      # Set configuration names
      names = data.get_legend()

      F = data.get_num_files()
      [x, y] = self.get_x_y(data)

      if(self._filename is not None):
         sys.stdout = open(self._filename, 'w')

      maxlength = 0
      for name in names:
         maxlength = max(maxlength, len(name))
      maxlength = str(maxlength)

      # Header line
      fmt = "%-" + maxlength + "s"
      lineDesc = data.get_axis_description_header()
      lineDescN = len(lineDesc) + 2
      lineDescFmt = "%-" + str(lineDescN) + "s |"
      print lineDescFmt % lineDesc,
      if(data.get_axis() == "threshold"):
         descs = self._thresholds
      else:
         descs = data.get_axis_descriptions()
      for name in names:
         print fmt % name,
      print ""

      # Loop over rows
      for i in range(0, len(x[0])):
         print lineDescFmt % descs[i],
         self._print_line(y[:, i], maxlength, "float")

      # Print stats
      for func in [verif.util.nanmin, verif.util.nanmean, verif.util.nanmax,
            verif.util.nanstd]:
         name = func.__name__[3:]
         print lineDescFmt % name,
         values = np.zeros(F, 'float')
         for f in range(0, F):
            values[f] = func(y[f, :])
         self._print_line(values, maxlength, "float")

      # Print count stats
      for func in [verif.util.nanmin, verif.util.nanmax]:
         name = func.__name__[3:]
         print lineDescFmt % ("num " + name),
         values = np.zeros(F, 'float')
         for f in range(0, F):
            values[f] = np.sum(y[f, :] == func(y, axis=0))
         self._print_line(values, maxlength, "int")

   def _csv_core(self, data):
      thresholds = self._thresholds

      data.set_axis(self._xaxis)

      # Set configuration names
      names = data.get_legend()

      F = data.get_num_files()
      [x, y] = self.get_x_y(data)

      if(self._filename is not None):
         sys.stdout = open(self._filename, 'w')

      # Header line
      header = data.get_axis_description_header(csv=True)
      for name in names:
         header = header + ',' + name
      print header

      # Loop over rows
      if(data.get_axis() == "threshold"):
         descs = self._thresholds
      else:
         descs = data.get_axis_descriptions(csv=True)
      for i in range(0, len(x[0])):
         line = str(descs[i])
         for j in range(0, len(y[:, i])):
            line = line + ',%g' % y[j, i]
         print line

   def _print_line(self, values, colWidth, type="float"):
      if(type == "int"):
         fmt = "%-" + colWidth + "i"
      else:
         fmt = "%-" + colWidth + ".2f"
      missfmt = "%-" + colWidth + "s"
      minI = np.argmin(values)
      maxI = np.argmax(values)
      for f in range(0, len(values)):
         value = values[f]
         if(np.isnan(value)):
            txt = missfmt % "--"
         else:
            txt = fmt % value
         if(minI == f):
            print verif.util.green(txt),
         elif(maxI == f):
            print verif.util.red(txt),
         else:
            print txt,
      print ""

   def _map_core(self, data):
      # Use the Basemap package if it is available
      hasBasemap = True
      try:
         from mpl_toolkits.basemap import Basemap
      except ImportError:
         verif.util.warning("Cannot load Basemap package")
         hasBasemap = False

      data.set_axis("location")
      labels = data.get_legend()
      F = data.get_num_files()
      lats = data.get_lats()
      lons = data.get_lons()
      ids = data.get_location_id()
      dlat = max(lats) - min(lats)
      dlon = max(lons) - min(lons)
      llcrnrlat = max(-90, min(lats) - dlat / 10)
      urcrnrlat = min(90, max(lats) + dlat / 10)
      llcrnrlon = min(lons) - dlon / 10
      urcrnrlon = max(lons) + dlon / 10
      if llcrnrlat > urcrnrlat - self._minLatLonRange:
         llcrnrlat = llcrnrlat - self._minLatLonRange/2.0
         urcrnrlat = urcrnrlat + self._minLatLonRange/2.0
      if llcrnrlon > urcrnrlon - self._minLatLonRange:
         llcrnrlon = llcrnrlon - self._minLatLonRange/2.0
         urcrnrlon = urcrnrlon + self._minLatLonRange/2.0

      # Check if we are wrapped across the dateline
      if(max(lons) - min(lons) > 180):
         minEastLon = min(lons[lons > 0])
         maxWestLon = max(lons[lons < 0])
         if(minEastLon - maxWestLon > 180):
            llcrnrlon = minEastLon - dlon / 10
            urcrnrlon = maxWestLon + dlon / 10 + 360
      if(self._xlim is not None):
         llcrnrlon = self._xlim[0]
         urcrnrlon = self._xlim[1]
      if(self._ylim is not None):
         llcrnrlat = self._ylim[0]
         urcrnrlat = self._ylim[1]

      res = verif.util.get_map_resolution(lats, lons)
      if(dlon < 5):
         dx = 1
      elif(dlon < 90):
         dx = 5
      else:
         dx = 10

      if(dlat < 5):
         dy = 1
      elif(dlat < 90):
         dy = 5
      else:
         dy = 10
      [x, y] = self.get_x_y(data)

      # Colorbar limits should be the same for all subplots
      clim = [verif.util.nanpercentile(y.flatten(), self._mapLowerPerc),
              verif.util.nanpercentile(y.flatten(), self._mapUpperPerc)]

      cmap = self._cmap

      # Forced limits
      if(self._clim is not None):
         clim = self._clim

      std = verif.util.nanstd(y)
      minDiff = std / 50

      for f in range(0, F):
         verif.util.subplot(f, F)
         if(self._mapType is not None and hasBasemap):
            if self._mapType == "simple":
               map = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                     urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, projection='mill',
                     resolution=res)
            else:
               # arcgisimage requires basemap to have an epsg option passed
               map = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                     urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, projection='mill',
                     resolution=res, epsg=4269)
            map.drawcoastlines(linewidth=0.25)
            map.drawcountries(linewidth=0.25)
            map.drawmapboundary()
            map.drawparallels(np.arange(-90., 120., dy), labels=[1, 0, 0, 0])
            map.drawmeridians(np.arange(-180., 420., dx), labels=[0, 0, 0, 1])
            map.fillcontinents(color='coral', lake_color='aqua', zorder=-1)
            x0, y0 = map(lons, lats)
            if self._mapType != "simple":
               if self._mapType == "sat":
                  service = 'ESRI_Imagery_World_2D'
               elif self._mapType == "topo":
                  service = 'World_Topo_Map'
               else:
                  service = self._mapType
               map.arcgisimage(service=service, xpixels=2000, verbose=True)
         else:
            # Use matplotlibs plotting functions, if we do not use Basemap
            map = mpl
            x0 = lons
            y0 = lats
            mpl.xlim([llcrnrlon, urcrnrlon])
            mpl.ylim([llcrnrlat, urcrnrlat])
         I = np.where(np.isnan(y[f, :]))[0]
         map.plot(x0[I], y0[I], 'kx')

         isMax = (y[f, :] == np.amax(y, 0)) &\
                 (y[f, :] > np.mean(y, 0) + minDiff)
         isMin = (y[f, :] == np.amin(y, 0)) &\
                 (y[f, :] < np.mean(y, 0) - minDiff)
         is_valid = (np.isnan(y[f, :]) == 0)
         s = self._ms*self._ms
         if(self._showRank):
            lmissing = None
            if(len(I) > 0):
               lmissing = map.scatter(x0[I], y0[I], s=s, c="k", marker="x")
            lsimilar = map.scatter(x0[is_valid], y0[is_valid], s=s, c="w")
            lmax = map.scatter(x0[isMax], y0[isMax], s=s, c="r")
            lmin = map.scatter(x0[isMin], y0[isMin], s=s, c="b")
         else:
            map.scatter(x0, y0, c=y[f, :], s=s, cmap=cmap)
            cb = map.colorbar()
            cb.set_label(self._metric.label(data))
            cb.set_clim(clim)
            mpl.clim(clim)
         if(self._mapLabelLocations):
            for i in range(0, len(x0)):
               value = y[f, i]

               if(not np.isnan(value)):
                  mpl.text(x0[i], y0[i], "%d %3.2f" % (ids[i], value))
         names = data.get_legend()
         if(self._title is not None):
            mpl.title(self._title)
         else:
            mpl.title(names[f])

      # Legend
      if(self._showRank):
         lines = [lmin, lsimilar, lmax]
         names = ["min", "similar", "max"]
         if lmissing is not None:
             lines.append(lmissing)
             names.append("missing")

         mpl.figlegend(lines, names, "lower center", ncol=4)


class Hist(Output):
   _reqThreshold = True
   _supThreshold = False

   def __init__(self, name):
      Output.__init__(self)
      self._name = name

      # Settings
      self._showPercent = True

   def get_x_y(self, data):
      F = data.get_num_files()
      allValues = [0] * F
      edges = self._thresholds
      for f in range(0, F):
         data.set_file_index(f)
         allValues[f] = data.get_scores(self._name)

      xx = (edges[0:-1] + edges[1:]) / 2
      y = np.zeros([F, len(xx)], 'float')
      x = np.zeros([F, len(xx)], 'float')
      for f in range(0, F):
         data.set_file_index(f)
         N = len(allValues[f][0])

         for i in range(0, len(xx)):
            if(i == len(xx) - 1):
               I = np.where((allValues[f][0] >= edges[i]) &
                            (allValues[f][0] <= edges[i + 1]))[0]
            else:
               I = np.where((allValues[f][0] >= edges[i]) &
                            (allValues[f][0] < edges[i + 1]))[0]
            y[f, i] = len(I) * 1.0
         x[f, :] = xx
      return [x, y]

   def _plot_core(self, data):
      data.set_axis("none")
      labels = data.get_legend()
      F = data.get_num_files()
      [x, y] = self.get_x_y(data)
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         if(self._showPercent):
            y[f] = y[f] * 1.0 / sum(y[f]) * 100
         mpl.plot(x[f], y[f], style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
      mpl.xlabel(data.get_axis_label("threshold"))
      if(self._showPercent):
         mpl.ylabel("Frequency (%)")
      else:
         mpl.ylabel("Frequency")
      mpl.grid()

   def _text_core(self, data):
      data.set_axis("none")
      labels = data.get_legend()

      F = data.get_num_files()
      [x, y] = self.get_x_y(data)

      if(self._filename is not None):
         sys.stdout = open(self._filename, 'w')

      maxlength = 0
      for label in labels:
         maxlength = max(maxlength, len(label))
      maxlength = str(maxlength)

      # Header line
      fmt = "%-" + maxlength + "s"
      lineDesc = data.get_axis_description_header()
      lineDescN = len(lineDesc) + 2
      lineDescFmt = "%-" + str(lineDescN) + "s |"
      print lineDescFmt % lineDesc,
      descs = self._thresholds
      for label in labels:
         print fmt % label,
      print ""

      # Loop over rows
      for i in range(0, len(x[0])):
         print lineDescFmt % descs[i],
         self._print_line(y[:, i], maxlength, "int")

      # Print count stats
      for func in [verif.util.nanmin, verif.util.nanmax]:
         name = func.__name__[3:]
         print lineDescFmt % ("num " + name),
         values = np.zeros(F, 'float')
         for f in range(0, F):
            values[f] = np.sum(y[f, :] == func(y, axis=0))
         self._print_line(values, maxlength, "int")

   def _csv_core(self, data):
      data.set_axis("none")
      labels = data.get_legend()

      F = data.get_num_files()
      [x, y] = self.get_x_y(data)

      if(self._filename is not None):
         sys.stdout = open(self._filename, 'w')

      maxlength = 0
      for label in labels:
         maxlength = max(maxlength, len(label))
      maxlength = str(maxlength)

      # Header line
      header = "threshold"
      for label in labels:
         header = header + "," + label
      print header

      # Loop over rows
      descs = self._thresholds
      for i in range(0, len(x[0])):
         line = str(descs[i])
         for j in range(0, len(y[:, i])):
            line = line + ",%g" % y[j, i]
         print line

   def _print_line(self, values, colWidth, type="float"):
      if(type == "int"):
         fmt = "%-" + colWidth + "i"
      else:
         fmt = "%-" + colWidth + ".2f"
      missfmt = "%-" + colWidth + "s"
      minI = np.argmin(values)
      maxI = np.argmax(values)
      for f in range(0, len(values)):
         value = values[f]
         if(np.isnan(value)):
            txt = missfmt % "--"
         else:
            txt = fmt % value
         if(minI == f):
            print verif.util.green(txt),
         elif(maxI == f):
            print verif.util.red(txt),
         else:
            print txt,
      print ""


class Sort(Output):
   _reqThreshold = False
   _supThreshold = False

   def __init__(self, name):
      Output.__init__(self)
      self._name = name

   def _plot_core(self, data):
      data.set_axis("none")
      labels = data.get_legend()
      F = data.get_num_files()
      for f in range(0, F):
         data.set_file_index(f)
         [x] = data.get_scores(self._name)
         x = np.sort(x)
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         y = np.linspace(0, 1, x.shape[0])
         mpl.plot(x, y, style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
      mpl.xlabel("Sorted " + data.get_axis_label("threshold"))
      mpl.grid()


class ObsFcst(Output):
   _supThreshold = False
   _description = "Plot observations and forecasts"

   def __init__(self):
      Output.__init__(self)

   def _plot_core(self, data):
      F = data.get_num_files()
      data.set_axis(self._xaxis)
      x = data.get_axis_values()

      isCont = data.is_axis_continuous()

      # Obs line
      mObs = verif.metric.Default("obs", aux="fcst")
      mObs.set_aggregator(self._aggregatorName)
      y = mObs.compute(data, None)
      self._plot_obs(x, y, isCont)

      mFcst = verif.metric.Default("fcst", aux="obs")
      mFcst.set_aggregator(self._aggregatorName)
      labels = data.get_legend()
      for f in range(0, F):
         data.set_file_index(f)
         color = self._get_color(f, F)
         style = self._get_style(f, F, isCont)

         y = mFcst.compute(data, None)
         mpl.plot(x, y, style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
      mpl.ylabel(data.get_variable_and_units())
      mpl.xlabel(data.get_axis_label())
      mpl.grid()
      if(data.is_axis_date()):
         mpl.gca().xaxis_date()
      else:
         mpl.gca().xaxis.set_major_formatter(data.get_axis_formatter())


class QQ(Output):
   _supThreshold = False
   _supX = False
   _description = "Quantile-quantile plot of obs vs forecasts"

   def __init__(self):
      Output.__init__(self)

   def get_x_y(self, data):
      x = list()
      y = list()
      F = len(data.get_filenames())
      for f in range(0, F):
         data.set_file_index(f)
         [xx, yy] = data.get_scores(["obs", "fcst"])
         x.append(np.sort(xx))
         y.append(np.sort(yy))
      return [x, y]

   def _plot_core(self, data):
      data.set_axis("none")
      data.set_index(0)
      labels = data.get_legend()
      F = data.get_num_files()
      [x, y] = self.get_x_y(data)
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)

         mpl.plot(x[f], y[f], style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
      mpl.ylabel("Sorted forecasts (" + data.get_units() + ")")
      mpl.xlabel("Sorted observations (" + data.get_units() + ")")
      ylim = list(mpl.ylim())
      xlim = list(mpl.xlim())
      axismin = min(min(ylim), min(xlim))
      axismax = max(max(ylim), max(xlim))
      self._plot_perfect_score([axismin, axismax], [axismin, axismax])
      mpl.grid()

   def _csv_core(self, data):
      data.set_axis("none")
      data.set_index(0)
      labels = data.get_legend()
      F = data.get_num_files()

      # Header
      header = ""
      for label in labels:
         header = header + label + "(obs)" + "," + label + "(fcst)"
      print header

      [x, y] = self.get_x_y(data)
      maxPairs = len(x[0])
      for f in range(1, F):
         maxPairs = max(maxPairs, len(x[f]))
      for i in range(0, maxPairs):
         line = ""
         for f in range(0, F):
            if(len(x[f]) < i):
               line = line + ","
            else:
               line = line + "%f,%f" % (x[f][i], y[f][i])
         print line


class Scatter(Output):
   _description = "Scatter plot of forecasts vs obs and lines showing quantiles of obs given forecast (use -r to specify)"
   _supThreshold = False
   _supX = False

   def __init__(self):
      Output.__init__(self)

   def _show_quantiles(self):
      return not self._simple

   def _plot_core(self, data):
      data.set_axis("none")
      data.set_index(0)
      labels = data.get_legend()
      F = data.get_num_files()
      for f in range(0, F):
         data.set_file_index(f)
         color = self._get_color(f, F)
         style = self._get_style(f, F, connectingLine=False)

         [x, y] = data.get_scores(["obs", "fcst"])
         alpha = 0.2
         if self._simple:
            alpha = 1
         mpl.plot(x, y, ".", color=color, label=labels[f], lw=self._lw, ms=self._ms, alpha=alpha)
         if(self._show_quantiles()):
            # Determine bin edges for computing quantiles
            # Use those provided by -r
            if(self._thresholds[0] is not None):
               edges = self._thresholds
            # For precip, we want a bin at exacly 0
            elif(re.compile("Precip.*").match(data.get_variable().name())):
               # Number of bins
               N = 10
               # The second to last edge should be such that we have at least
               # Nmin data points above
               Nmin = 50.0
               pUpper = 100.0 - Nmin / y.shape[0] * 100.0
               # But no lower than 90th percentile, incase we don't have very many values
               if(pUpper < 90):
                  pUpper = 90
               edges = np.linspace(0, 50, 21)
               edges = np.array([0, 0.001, 1, 2, 3, 5, 7, 10, 13, 16, 20, 25, 30, 35, 40, 45, 50])
               edges = np.append(np.array([0]), np.linspace(0.001, np.percentile(y, pUpper), N - 1))
               edges = np.append(edges, np.array([np.max(y)]))
            # Regular variables
            else:
               # How many quantile boxes should we make?
               N = max(8, min(30, x.shape[0] / 100))

               # We want the lower bin to cointain at least 50 points, so find
               # which percentile will give us 50 points
               Nmin = 50.0
               pLower = Nmin / y.shape[0] * 100.0
               # If we don't have very much data, then use an upper bound of 10%tile
               if(pLower > 10):
                  pLower = 10
               pUpper = 100.0 - pLower
               # Create evenly spaced values from the point where we have at 50
               # below to 50 above
               edges = np.linspace(np.percentile(y, pLower), np.percentile(y, pUpper), N)
               # Add on the end points
               edges = np.append(np.array([np.min(y)]), edges)
               edges = np.append(edges, np.array([np.max(y)]))

            bins = (edges[1:] + edges[0:-1])/2

            # What quantile lines should be drawn?
            quantiles = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
            values = np.nan*np.zeros([len(quantiles), len(bins)], 'float')
            for q in range(0, len(quantiles)):
               for i in range(0, len(bins)):
                  I = np.where((y >= edges[i]) & (y < edges[i+1]))[0]
                  if(len(I) > 0):
                     values[q, i] = np.percentile(x[I], quantiles[q]*100)
               style = 'k-'
               lw = 2
               if(q == 0 or q == len(quantiles)-1):
                  style = 'ko--'
               elif(q == (len(quantiles)-1)/2):
                  style = 'ko-'
                  lw = 4
               # Write labels for the quantile lines, but only do it for one file
               label = ""
               if(f == 0):
                  if(q == 0 or q == len(quantiles) - 1):
                     label = "%d%%" % (quantiles[q] * 100)
                  # Instead of writing all labels, only summarize the middle ones
                  elif(q == 1 and len(quantiles) > 3):
                     label = "%d%%-%d%%" % (quantiles[1] * 100, (quantiles[len(quantiles) - 2] * 100))
                  elif(q == 1 and len(quantiles) == 3):
                     label = "%d%%" % (quantiles[1] * 100)
               mpl.plot(values[q, :], bins, style, lw=lw, alpha=0.5, label=label)
            for i in range(0, len(bins)):
               mpl.plot([values[0, i], values[-1, i]], [bins[i], bins[i]], 'k-')
      mpl.ylabel("Forecasts (" + data.get_units() + ")")
      mpl.xlabel("Observations (" + data.get_units() + ")")
      ylim = mpl.ylim()
      xlim = mpl.xlim()
      axismin = min(min(ylim), min(xlim))
      axismax = max(max(ylim), max(xlim))
      mpl.plot([axismin, axismax], [axismin, axismax], "--",
            color=[0.3, 0.3, 0.3], lw=3, zorder=-100)
      mpl.grid()
      mpl.gca().set_aspect(1)


class Change(Output):
   _supThreshold = False
   _supX = False
   _description = "Forecast skill (MAE) as a function of change in obs from "\
                  "previous day"

   def __init__(self):
      Output.__init__(self)

   def _plot_core(self, data):
      data.set_axis("all")
      data.set_index(0)
      labels = data.get_legend()
      # Find range
      data.set_file_index(0)
      [obs, fcst] = data.get_scores(["obs", "fcst"])
      change = obs[1:, Ellipsis] - obs[0:-1, Ellipsis]
      maxChange = np.nanmax(abs(change.flatten()))
      edges = np.linspace(-maxChange, maxChange, 20)
      bins = (edges[1:] + edges[0:-1]) / 2
      F = data.get_num_files()

      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         data.set_file_index(f)
         [obs, fcst] = data.get_scores(["obs", "fcst"])
         change = obs[1:, Ellipsis] - obs[0:-1, Ellipsis]
         err = abs(obs - fcst)
         err = err[1:, Ellipsis]
         x = np.nan * np.zeros(len(bins), 'float')
         y = np.nan * np.zeros(len(bins), 'float')

         for i in range(0, len(bins)):
            I = (change > edges[i]) & (change <= edges[i + 1])
            y[i] = verif.util.nanmean(err[I])
            x[i] = verif.util.nanmean(change[I])
         mpl.plot(x, y, style, color=color, lw=self._lw, ms=self._ms,
               label=labels[f])
      self._plot_perfect_score(x, 0)
      mpl.xlabel("Daily obs change (" + data.get_units() + ")")
      mpl.ylabel("MAE (" + data.get_units() + ")")
      mpl.grid()


class Cond(Output):
   _description = "Plots forecasts as a function of obs (use -r to specify "\
                  "bin-edges)"
   _default_axis = "threshold"
   _default_bin_type = "within"
   _reqThreshold = True
   _supThreshold = True
   _supX = False

   def supports_threshold(self):
      return True

   def _plot_core(self, data):
      data.set_axis("none")
      data.set_index(0)
      [lowerT, upperT, x] = self._get_threshold_limits(self._thresholds)

      labels = data.get_legend()
      F = data.get_num_files()
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         data.set_file_index(f)

         of = np.zeros(len(x), 'float')
         fo = np.zeros(len(x), 'float')
         xof = np.zeros(len(x), 'float')
         xfo = np.zeros(len(x), 'float')
         mof = verif.metric.Conditional("obs", "fcst", np.mean)  # F | O
         mfo = verif.metric.Conditional("fcst", "obs", np.mean)  # O | F
         xmof = verif.metric.XConditional("obs", "fcst")  # F | O
         xmfo = verif.metric.XConditional("fcst", "obs")  # O | F
         mof0 = verif.metric.Conditional("obs", "fcst", np.mean)  # F | O
         for i in range(0, len(lowerT)):
            fo[i] = mfo.compute(data, [lowerT[i], upperT[i]])
            of[i] = mof.compute(data, [lowerT[i], upperT[i]])
            xfo[i] = xmfo.compute(data, [lowerT[i], upperT[i]])
            xof[i] = xmof.compute(data, [lowerT[i], upperT[i]])
         mpl.plot(xof, of, style, color=color, label=labels[f] + " (F|O)",
               lw=self._lw, ms=self._ms)
         mpl.plot(fo, xfo, style, color=color, label=labels[f] + " (O|F)",
               lw=self._lw, ms=self._ms, alpha=0.5)
      mpl.ylabel("Forecasts (" + data.get_units() + ")")
      mpl.xlabel("Observations (" + data.get_units() + ")")
      ylim = mpl.ylim()
      xlim = mpl.xlim()
      axismin = min(min(ylim), min(xlim))
      axismax = max(max(ylim), max(xlim))
      self._plot_perfect_score([axismin, axismax], [axismin, axismax])
      mpl.grid()
      mpl.gca().set_aspect(1)


class SpreadSkill(Output):
   _supThreshold = False
   _supX = False
   _description = "Spread/skill plot showing RMSE of ensemble mean as a function of ensemble spread (use -r to specify spread thresholds)"

   def __init__(self):
      Output.__init__(self)

   def _plot_core(self, data):
      data.set_axis("all")
      data.set_index(0)
      labels = data.get_legend()
      F = data.get_num_files()
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F, connectingLine=False)
         data.set_file_index(f)

         data.set_file_index(f)
         [obs, fcst, spread] = data.get_scores(["obs", "fcst", "spread"])
         spread = spread.flatten()
         skill = (obs.flatten() - fcst.flatten())**2
         x = np.nan*np.zeros(len(self._thresholds), 'float')
         y = np.nan*np.zeros(len(x), 'float')
         for i in range(1, len(self._thresholds)):
            I = np.where((np.isnan(spread) == 0) &
                         (np.isnan(skill) == 0) &
                         (spread > self._thresholds[i - 1]) &
                         (spread <= self._thresholds[i]))[0]
            if(len(I) > 0):
               x[i] = np.mean(spread[I])
               y[i] = np.sqrt(np.mean(skill[I]))

         style = self._get_style(f, F)
         mpl.plot(x, y, style, color=color, lw=self._lw, ms=self._ms, label=labels[f])
      ylim = list(mpl.ylim())
      xlim = list(mpl.xlim())
      ylim[0] = 0
      xlim[0] = 0
      axismin = min(min(ylim), min(xlim))
      axismax = max(max(ylim), max(xlim))
      mpl.xlim(xlim)
      mpl.ylim(ylim)
      self._plot_perfect_score([axismin, axismax], [axismin, axismax])
      mpl.xlabel("Spread (" + data.get_units() + ")")
      mpl.ylabel("RMSE (" + data.get_units() + ")")
      mpl.grid()


class Count(Output):
   _description = "Counts number of forecasts above or within thresholds "\
         "(use -r to specify bin-edges). Use -binned to count number in "\
         "bins, nstead of number above each threshold."
   _default_axis = "threshold"
   _default_bin_type = "within"
   _reqThreshold = True
   _supThreshold = True
   _supX = False

   def _plot_core(self, data):
      data.set_axis("none")
      data.set_index(0)
      [lowerT, upperT, x] = self._get_threshold_limits(self._thresholds)

      labels = data.get_legend()
      F = data.get_num_files()
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         data.set_file_index(f)

         Nobs = np.zeros(len(x), 'float')
         Nfcst = np.zeros(len(x), 'float')
         obs = verif.metric.Count("obs")
         fcst = verif.metric.Count("fcst")
         for i in range(0, len(lowerT)):
            Nobs[i] = obs.compute(data, [lowerT[i], upperT[i]])
            Nfcst[i] = fcst.compute(data, [lowerT[i], upperT[i]])
         mpl.plot(x, Nfcst, style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
      self._plot_obs(x, Nobs)
      mpl.ylabel("Number")
      mpl.xlabel(data.get_axis_label())
      mpl.grid()


class TimeSeries(Output):
   _description = "Plot observations and forecasts as a time series "\
         "(i.e. by concatinating all offsets). '-x <dimension>' has no "\
         "effect, as it is always shown by date."
   _supThreshold = False
   _supX = False

   def _plot_core(self, data):
      F = data.get_num_files()
      data.set_axis("all")
      dates = data.get_axis_values("date")
      offsets = data.get_axis_values("offset")

      # Connect the last offset of a day with the first offset on the next day
      # This only makes sense if the obs/fcst don't span more than a day
      connect = min(offsets) + 24 > max(offsets)
      minOffset = min(offsets)

      obs = data.get_scores("obs")[0]
      for d in range(0, obs.shape[0]):
         # Obs line
         x = dates[d] + offsets / 24.0
         y = verif.util.nanmean(obs[d, :, :], axis=1)
         if(connect and d < obs.shape[0] - 1):
            obsmean = verif.util.nanmean(obs[d + 1, 0, :], axis=0)
            x = np.insert(x, x.shape[0], dates[d + 1] + minOffset / 24.0)
            y = np.insert(y, y.shape[0], obsmean)

         if(d == 0):
            xmin = np.min(x)
         elif(d == obs.shape[0] - 1):
            xmax = np.max(x)

         lab = "obs" if d == 0 else ""
         mpl.rcParams['ytick.major.pad'] = '20'
         mpl.rcParams['xtick.major.pad'] = '20'
         mpl.plot(x, y, ".-", color=[0.3, 0.3, 0.3], lw=5, label=lab)

         # Forecast lines
         labels = data.get_legend()
         for f in range(0, F):
            data.set_file_index(f)
            color = self._get_color(f, F)
            style = self._get_style(f, F)

            fcst = data.get_scores("fcst")[0]
            x = dates[d] + offsets / 24.0
            y = verif.util.nanmean(fcst[d, :, :], axis=1)
            if(connect and d < obs.shape[0] - 1):
               x = np.insert(x, x.shape[0], dates[d + 1] + minOffset / 24.0)
               y = np.insert(y, y.shape[0], verif.util.nanmean(fcst[d + 1, 0, :]))
            lab = labels[f] if d == 0 else ""
            mpl.rcParams['ytick.major.pad'] = '20'
            mpl.rcParams['xtick.major.pad'] = '20'
            mpl.plot(x, y, style, color=color, lw=self._lw, ms=self._ms,
                  label=lab)

      mpl.xlabel(data.get_axis_label("date"))
      if(self._ylabel is None):
         mpl.ylabel(data.get_variable_and_units())
      else:
         mpl.ylabel(self._ylabel)
      mpl.grid()
      mpl.gca().xaxis.set_major_formatter(data.get_axis_formatter("date"))

      if(self._tight):
         oldTicks = mpl.gca().get_xticks()
         diff = oldTicks[1] - oldTicks[0]  # keep auto tick interval
         tickRange = np.arange(round(xmin), round(xmax) + diff, diff)
         # make new ticks, to start from the first day of the desired interval
         mpl.gca().set_xticks(tickRange)
         mpl.autoscale(enable=True, axis=u'x', tight=True)  # make xaxis tight


class Meteo(Output):
   _description = "Plot a meteogram, with deterministic forecast, all quantile lines available, and observations. If multiple dates and stations are used, then the average is made."
   _supThreshold = False
   _supX = False
   _obsCol = [1, 0, 0]
   _fcstCol = [0, 1, 0]

   def _plot_core(self, data):
      F = data.get_num_files()
      data.set_axis("all")
      dates = data.get_axis_values("date")
      offsets = data.get_axis_values("offset")
      x = dates[0] + offsets/24.0
      isSingleDate = len(dates) == 1

      # Plot obs line
      obs = data.get_scores("obs")[0]
      obs = verif.util.nanmean(verif.util.nanmean(obs, axis=0), axis=1)
      mpl.plot(x, obs, "o-", color=self._obsCol, lw=2, ms=8, label="Observations")

      # Plot deterministic forecast
      fcst = data.get_scores("fcst")[0]
      fcst = verif.util.nanmean(verif.util.nanmean(fcst, axis=0), axis=1)
      mpl.plot(x, fcst, "o-", color=self._fcstCol, lw=2, ms=8, label="Deterministic")

      # Plot quantiles
      quantiles = data.get_quantiles()
      quantiles = np.sort(quantiles)
      y = np.zeros([len(offsets), len(quantiles)], 'float')
      for i in range(0, len(quantiles)):
         quantile = quantiles[i]/100
         var = data.get_q_var(quantile)
         y[:, i] = verif.util.nanmean(verif.util.nanmean(data.get_scores(var)[0], axis=0), axis=1)
      for i in range(0, len(quantiles)):
         style = "k-"
         if(i == 0 or i == len(quantiles) - 1):
            style = "k--"
         label = "%d%%" % (quantiles[i])
         mpl.plot(x, y[:, i], style, label=label, zorder=-1)

      # Fill areas betweeen lines
      Ncol = (len(quantiles)-1)/2
      for i in range(0, Ncol):
         color = [(1 - (i + 0.0) / Ncol)] * 3
         verif.util.fill(x, y[:, i], y[:, len(quantiles) - 1 - i], color,
               zorder=-2)

      # Labels and ticks
      if(self._ylabel is None):
         mpl.ylabel(data.get_variable_and_units())
      else:
         mpl.ylabel(self._ylabel)
      mpl.grid()
      if(data.is_axis_date()):
         mpl.gca().xaxis_date()
      else:
         mpl.gca().xaxis.set_major_formatter(data.get_axis_formatter())

      if(np.min(x) == np.max(x)):
         mpl.xlim(x[0], x[0] + 1)
      else:
         mpl.xlim(np.min(x), np.max(x))
      mpl.gca().xaxis.set_major_locator(mpldates.DayLocator(interval=1))
      mpl.gca().xaxis.set_minor_locator(mpldates.HourLocator(interval=6))
      mpl.gca().xaxis.set_major_formatter(mpldates.DateFormatter('\n  %a %d %b %Y'))
      mpl.gca().xaxis.set_minor_formatter(mpldates.DateFormatter('%H'))

      # Hour labels
      minlabels = [tick.label1 for tick in mpl.gca().xaxis.get_minor_ticks()]
      for i in minlabels:
         i.set_fontsize(self._tickfs)

      # Date labels
      majlabels = [tick.label1 for tick in mpl.gca().xaxis.get_major_ticks()]
      for i in range(0, len(majlabels)):
         label = majlabels[i]
         if(isSingleDate and i < len(majlabels)-1):
            label.set_horizontalalignment('left')
            label.set_verticalalignment('top')
            label.set_fontsize(self._tickfs)
            # Moves major labels to the top of the graph. The x-coordinate
            # seems to be irrelevant. When y-coord is 1, the label is near the
            # top. For 1.1 it is above the graph
            label.set_position((0, -0.035))
         else:
            # Turn off the last date label, since it is outside the graph
            label.set_visible(0)
      if(not isSingleDate):
         mpl.xlabel("Time of day (h)")

      mpl.gca().xaxis.grid(True, which='major', color='k', zorder=-10, linestyle='-', linewidth=2)
      mpl.gca().xaxis.grid(True, which='minor', color='k', zorder=0, linestyle='--')
      mpl.gca().yaxis.grid(True, which='major', color='k', zorder=0)


class PitHist(Output):
   _description = "Histogram of PIT values"
   _supThreshold = False
   _supX = False

   def __init__(self, metric):
      Output.__init__(self)
      self._numBins = 10
      self._metric = metric

   def _show_stats(self):
      return not self._simple

   def _show_expected_line(self):
      return not self._simple

   def _legend(self, data, names=None):
      pass

   def _plot_core(self, data):
      F = data.get_num_files()
      labels = data.get_legend()
      for f in range(0, F):
         verif.util.subplot(f, F)
         color = self._get_color(f, F)
         data.set_axis("none")
         data.set_index(0)
         data.set_file_index(f)
         pit = self._metric.compute(data, None)

         width = 1.0 / self._numBins
         x = np.linspace(0, 1, self._numBins + 1)
         N = np.histogram(pit, x)[0]
         n = N * 1.0 / sum(N)
         color = "gray"
         xx = x[range(0, len(x) - 1)]
         mpl.bar(xx, n * 100.0, width=width, color=color)
         mpl.plot([0, 1], [100.0 / self._numBins, 100.0 / self._numBins],
               'k--')
         mpl.title(labels[f])
         ytop = 200.0 / self._numBins
         mpl.gca().set_ylim([0, ytop])
         if(f == 0):
            mpl.ylabel("Frequency (%)")
         else:
            mpl.gca().set_yticks([])

         if(self._show_expected_line()):
            # Multiply by 100 to get to percent
            std = verif.metric.PitDev.deviation_std(pit, self._numBins) * 100

            mpl.plot([0, 1], [100.0 / self._numBins - 2 * std,
               100.0 / self._numBins - 2 * std], "r-")
            mpl.plot([0, 1], [100.0 / self._numBins + 2 * std,
               100.0 / self._numBins + 2 * std], "r-")
            lower = [100.0 / self._numBins - 2 * std,
                  100.0 / self._numBins - 2 * std]
            upper = [100.0 / self._numBins + 2 * std,
                  100.0 / self._numBins + 2 * std]
            verif.util.fill([0, 1], lower, upper, "r", zorder=100, alpha=0.5)

         # Compute calibration deviation
         if(self._show_stats()):
            D = verif.metric.PitDev.deviation(pit, self._numBins)
            D0 = verif.metric.PitDev.expected_deviation(pit, self._numBins)
            ign = verif.metric.PitDev.ignorance_potential(pit, self._numBins)
            mpl.text(0, mpl.ylim()[1], "Dev: %2.4f\nExp: %2.4f\nIgn: %2.4f"
                  % (D, D0, ign), verticalalignment="top")

         mpl.xlabel("Cumulative probability")


class Discrimination(Output):
   _description = "Discrimination diagram for a certain threshold (-r)"
   _reqThreshold = True
   _supX = False

   def __init__(self):
      Output.__init__(self)
      self._numBins = 10

   def _plot_core(self, data):
      labels = data.get_legend()

      F = data.get_num_files()

      data.set_axis("none")
      data.set_index(0)
      data.set_file_index(0)
      mpl.bar(np.nan, np.nan, color="w", ec="k", lw=self._lw, label="Observed")
      mpl.bar(np.nan, np.nan, color="k", ec="k", lw=self._lw, label="Not observed")
      for t in range(0, len(self._thresholds)):
         threshold = self._thresholds[t]
         var = data.get_p_var(threshold)
         [obs, p] = data.get_scores(["obs", var])

         # Determine the number of bins to use # (at least 11, at most 25)
         edges = np.linspace(0, 1, self._numBins + 1)

         y1 = np.nan * np.zeros([F, len(edges) - 1], 'float')
         y0 = np.nan * np.zeros([F, len(edges) - 1], 'float')
         n = np.zeros([F, len(edges) - 1], 'float')
         for f in range(0, F):
            color = self._get_color(f, F)
            style = self._get_style(f, F)
            data.set_file_index(f)
            data.set_axis("none")
            data.set_index(0)
            var = data.get_p_var(threshold)
            [obs, p] = data.get_scores(["obs", var])

            if(self._binType == "below"):
               p = p
               obs = obs < threshold
            elif(self._binType == "above"):
               p = 1 - p
               obs = obs > threshold
            else:
               verif.util.error("Bin type must be one of 'below' or"
                     "'above' for discrimination diagram")

            clim = np.mean(obs)
            I1 = np.where(obs == 1)[0]
            I0 = np.where(obs == 0)[0]
            # Compute frequencies
            for i in range(0, len(edges) - 1):
               y0[f, i] = np.mean((p[I0] >= edges[i]) & (p[I0] < edges[i + 1]))
               y1[f, i] = np.mean((p[I1] >= edges[i]) & (p[I1] < edges[i + 1]))
            label = labels[f]
            if(not t == 0):
               label = ""
            # Figure out where to put the bars. Each file will have pairs of
            # bars, so try to space them nicely.
            width = 1.0 / self._numBins
            space = 1.0 / self._numBins * 0.2
            shift = (0.5 / self._numBins - width)
            center = (edges[0:-1]+edges[1:])/2
            clustercenter = edges[0:-1] + 1.0*(f + 1) / (F + 1) * width
            clusterwidth = width * 0.8 / F
            barwidth = clusterwidth / 2
            shift = barwidth
            mpl.bar(clustercenter-shift, y1[f, :], barwidth, color=color,
                  ec=color, lw=self._lw, label=label)
            mpl.bar(clustercenter, y0[f, :], barwidth, color="w", ec=color,
                  lw=self._lw)
         mpl.plot([clim, clim], [0, 1], "k-")

      mpl.xlim([0, 1])
      mpl.xlabel("Forecasted probability")
      mpl.ylabel("Frequency")
      units = " " + data.get_units()
      if(self._binType == "below"):
         mpl.title("Discrimination diagram for obs < " + str(threshold) + units)
      elif(self._binType == "above"):
         mpl.title("Discrimination diagram for obs > " + str(threshold) + units)
      else:
         verif.util.error("Bin type must be one of 'below' or"
               "'above' for discrimination diagram")


class Reliability(Output):
   _description = "Reliability diagram for a certain threshold (-r)"
   _reqThreshold = True
   _supX = False
   _legLoc = "lower right"

   def __init__(self):
      Output.__init__(self)
      self._minCount = 5  # Min number of valid data points to show in graph

   def _show_count(self):
      return not self._simple

   def _shade_confidence(self):
      return not self._simple

   def _shade_no_skill(self):
      return not self._simple

   def _plot_core(self, data):
      labels = data.get_legend()

      F = data.get_num_files()
      ax = mpl.gca()
      if(self._show_count()):
         axi = mpl.axes([0.3, 0.65, 0.2, 0.2])
      mpl.sca(ax)

      data.set_axis("none")
      data.set_index(0)
      data.set_file_index(0)
      for t in range(0, len(self._thresholds)):
         threshold = self._thresholds[t]
         var = data.get_p_var(threshold)
         [obs, p] = data.get_scores(["obs", var])

         # Determine the number of bins to use # (at least 11, at most 25)
         N = min(25, max(11, int(len(obs) / 1000)))
         N = 11
         edges = np.linspace(0, 1, N + 1)
         edges = np.array([0, 0.05, 0.15, 0.25, 0.35, 0.45,
            0.55, 0.65, 0.75, 0.85, 0.95, 1])
         x = np.zeros([len(edges) - 1, F], 'float')

         y = np.nan * np.zeros([F, len(edges) - 1], 'float')
         n = np.zeros([F, len(edges) - 1], 'float')
         v = np.zeros([F, len(edges) - 1], 'float')  # Variance
         # Draw reliability lines
         for f in range(0, F):
            color = self._get_color(f, F)
            style = self._get_style(f, F)
            data.set_file_index(f)
            data.set_axis("none")
            data.set_index(0)
            var = data.get_p_var(threshold)
            [obs, p] = data.get_scores(["obs", var])

            if(self._binType == "below"):
               p = p
               obs = obs < threshold
            elif(self._binType == "above"):
               p = 1 - p
               obs = obs > threshold
            else:
               verif.util.error("Bin type must be one of 'below' or"
                     "'above' for reliability plot")

            clim = np.mean(obs)
            # Compute frequencies
            for i in range(0, len(edges) - 1):
               q = (p >= edges[i]) & (p < edges[i + 1])
               I = np.where(q)[0]
               if(len(I) > 0):
                  n[f, i] = len(obs[I])
                  # Need at least 10 data points to be valid
                  if(n[f, i] >= self._minCount):
                     y[f, i] = np.mean(obs[I])
                     v[f, i] = np.var(obs[I])
                  x[i, f] = np.mean(p[I])

            label = labels[f]
            if(not t == 0):
               label = ""
            mpl.plot(x[:, f], y[f], style, color=color, lw=self._lw,
                  ms=self._ms, label=label)

         # Draw confidence bands (do this separately so that these lines don't
         # sneak into the legend)
         for f in range(0, F):
            color = self._get_color(f, F)
            if(self._shade_confidence()):
               self._plot_confidence(x[:, f], y[f], v[f], n[f], color=color)

         # Draw lines in inset diagram
         if(self._show_count()):
            if(np.max(n) > 1):
               for f in range(0, F):
                  color = self._get_color(f, F)
                  axi.plot(x[:, f], n[f], style, color=color, lw=self._lw,
                        ms=self._ms * 0.75)
               axi.xaxis.set_major_locator(mpl.NullLocator())
               axi.set_yscale('log')
               axi.set_title("Number")
               axi.grid('on')
      mpl.sca(ax)
      self._plot_obs([0, 1], [0, 1], label="")
      mpl.axis([0, 1, 0, 1])
      color = "gray"
      # Climatology line
      mpl.plot([0, 1], [clim, clim], "--", color=color, label="")
      mpl.plot([clim, clim], [0, 1], "--", color=color)
      # No-skill line
      mpl.plot([0, 1], [clim / 2, 1 - (1 - clim) / 2], "--", color=color)
      if(self._shade_no_skill()):
         verif.util.fill([clim, 1], [0, 0], [clim, 1 - (1 - clim) / 2],
               col=[1, 1, 1], zorder=-100, hatch="\\")
         verif.util.fill([0, clim], [clim / 2, clim, 0], [1, 1],
               col=[1, 1, 1], zorder=-100, hatch="\\")
      mpl.xlabel("Forecasted probability")
      mpl.ylabel("Observed frequency")
      units = " " + data.get_units()
      if(self._binType == "below"):
         mpl.title("Reliability diagram for obs < " + str(threshold) + units)
      elif(self._binType == "above"):
         mpl.title("Reliability diagram for obs > " + str(threshold) + units)
      else:
         verif.util.error("Bin type must be one of 'below' or"
               "'above' for reliability plot")
      mpl.gca().set_aspect(1)


class IgnContrib(Output):
   _description = "Binary Ignorance contribution diagram for a single "\
         "threshold (-r). Shows how much each probability issued contributes "\
         "to the total ignorance."
   _reqThreshold = True
   _supX = False
   _legLoc = "upper center"
   _experimental = True

   def __init__(self):
      Output.__init__(self)

   def _plot_core(self, data):
      labels = data.get_legend()

      if(len(self._thresholds) != 1):
         verif.util.error("IgnContrib diagram requires exactly one threshold")
      threshold = self._thresholds[0]

      F = data.get_num_files()

      mpl.subplot(2, 1, 1)
      units = " " + data.get_units()
      titlestr = "Ignorance contribution diagram for obs > " +\
                 str(self._thresholds[0]) + units
      mpl.title(titlestr)

      data.set_axis("none")
      data.set_index(0)
      data.set_file_index(0)
      mpl.subplot(2, 1, 1)
      var = data.get_p_var(threshold)
      [obs, p] = data.get_scores(["obs", var])

      # Determine the number of bins to use # (at least 11, at most 25)
      N = min(25, max(11, int(len(obs) / 1000)))
      edges = np.linspace(0, 1, N + 1)

      x = np.zeros([F, len(edges) - 1], 'float')
      y = np.nan * np.zeros([F, len(edges) - 1], 'float')
      n = np.zeros([F, len(edges) - 1], 'float')

      # Draw reliability lines
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         data.set_file_index(f)
         data.set_axis("none")
         data.set_index(0)
         var = data.get_p_var(threshold)
         [obs, p] = data.get_scores(["obs", var])

         if(self._binType == "below"):
            p = p
            obs = obs < threshold
         elif(self._binType == "above"):
            p = 1 - p
            obs = obs > threshold
         else:
            verif.util.error("Bin type must be one of 'below' or 'above' "
                         "for igncontrib plot")

         clim = np.mean(obs)
         # Compute frequencies
         for i in range(0, len(edges) - 1):
            q = (p >= edges[i]) & (p < edges[i + 1])
            I = np.where(q)[0]
            if(len(I) > 0):
               n[f, i] = len(obs[I])
               x[f, i] = np.mean(p[I])
               # Need at least 10 data points to be valid
               if(n[f, i] >= 1):
                  I0 = np.where(obs[I] == 0)
                  I1 = np.where(obs[I] == 1)
                  y[f, i] = -np.sum(np.log2(p[I[I1]])) -\
                             np.sum(np.log2(1 - p[I[I0]]))

         label = labels[f]
         mpl.plot(x[f], y[f] / np.sum(n[f]) * len(n[f]), style, color=color,
               lw=self._lw, ms=self._ms, label=label)
      mpl.ylabel("Ignorance contribution")

      # Draw expected sharpness
      xx = np.linspace(0, 1, 100)
      yy = -(xx * np.log2(xx) + (1 - xx) * np.log2(1 - xx))
      mpl.plot(xx, yy, "--", color="gray")
      yy = -np.log2(clim) * np.ones(len(xx))

      # Show number in each bin
      mpl.subplot(2, 1, 2)
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         mpl.plot(x[f], n[f], style, color=color, lw=self._lw, ms=self._ms)
      mpl.xlabel("Forecasted probability")
      mpl.ylabel("N")

      # Switch back to top subpplot, so the legend works
      mpl.subplot(2, 1, 1)


class EconomicValue(Output):
   _description = "Economic value diagram for a single "\
         "threshold (-r). Shows what fraction of costs/loses can be reduced by"\
         " the forecast relative to using climatology."
   _reqThreshold = True
   _supX = False
   _experimental = True

   def __init__(self):
      Output.__init__(self)

   def _plot_core(self, data):
      labels = data.get_legend()

      if(len(self._thresholds) != 1):
         verif.util.error("Economic value diagram requires exactly one threshold")
      threshold = self._thresholds[0]

      F = data.get_num_files()

      units = " " + data.get_units()
      if(self._binType == "below"):
         mpl.title("Economic value for obs < " + str(self._thresholds[0]) + units)
      elif(self._binType == "above"):
         mpl.title("Economic value for obs > " + str(self._thresholds[0]) + units)
      else:
         verif.util.error("Bin type must be one of 'below' or"
               "'above' for reliability plot")

      data.set_axis("none")
      data.set_index(0)
      data.set_file_index(0)
      var = data.get_p_var(threshold)
      [obs, p] = data.get_scores(["obs", var])

      # Determine the number of bins to use # (at least 11, at most 25)
      N = min(25, max(11, int(len(obs) / 1000)))
      N = 20
      costLossRatios = np.linspace(0, 1, N + 1)
      # import scipy.stats
      # costLossRatios = scipy.stats.norm(0,1).cdf(np.linspace(-5,5,N))
      costLossRatios = np.linspace(0, 1, N + 1)**3

      x = costLossRatios
      y = np.nan * np.zeros([F, len(costLossRatios)], 'float')
      n = np.zeros([F, len(costLossRatios)], 'float')

      # Draw reliability lines
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         data.set_file_index(f)
         data.set_axis("none")
         data.set_index(0)
         var = data.get_p_var(threshold)
         [obs, p] = data.get_scores(["obs", var])

         if(self._binType == "below"):
            p = p
            obs = obs < threshold
         elif(self._binType == "above"):
            p = 1 - p
            obs = obs > threshold
         else:
            verif.util.error("Bin type must be one of 'below' or 'above' " "for economicvalue plot")

         clim = np.mean(obs)
         # Compute frequencies
         for i in range(0, len(costLossRatios)):
            costLossRatio = costLossRatios[i]
            Icost = np.where(p >= costLossRatio)[0]
            Iloss = np.where((p < costLossRatio) & (obs == 1))[0]
            loss = 1
            cost = costLossRatio * loss
            totalCost = cost * len(Icost) + loss * len(Iloss)
            totalCost = totalCost / len(obs)
            # Cost when using a climatological forecast
            climCost = min(clim * loss, cost)
            perfectCost = clim * cost
            economicValue = 0
            if(climCost != perfectCost):
               economicValue = (climCost-totalCost) / (climCost - perfectCost)
            y[f, i] = economicValue

         label = labels[f]
         mpl.plot(costLossRatios, y[f], style, color=color,
               lw=self._lw, ms=self._ms, label=label)
      mpl.xlabel("Cost-loss ratio")
      mpl.ylabel("Economic value")
      mpl.xlim([0, 1])
      mpl.ylim([0, 1])
      mpl.grid()


class Roc(Output):
   _description = "Plots the receiver operating characteristics curve for a single threshold (-r)"
   _supX = False
   _reqThreshold = True

   def __init__(self):
      Output.__init__(self)
      self._labelQuantiles = True

   def _plot_core(self, data):
      threshold = self._thresholds[0]   # Observation threshold
      if(threshold is None):
         verif.util.error("Roc plot needs a threshold (use -r)")

      quantiles = list(data.get_quantiles())
      if(len(quantiles) == 0):
         verif.util.error("Your files do not have any quantiles")

      F = data.get_num_files()
      labels = data.get_legend()
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         data.set_axis("none")
         data.set_index(0)
         data.set_file_index(f)
         scores = data.get_scores(["obs"] + data.get_quantile_names())
         obs = scores[0]
         fcsts = scores[1:]
         y = np.nan * np.zeros([len(quantiles)], 'float')
         x = np.nan * np.zeros([len(quantiles)], 'float')
         for i in range(0, len(quantiles)):
            # Compute the hit rate and false alarm rate by using the given
            # quantile from the distribution as the forecast
            fcst = fcsts[i]
            a = np.ma.sum((fcst >= threshold) & (obs >= threshold))  # Hit
            b = np.ma.sum((fcst >= threshold) & (obs < threshold))   # FA
            c = np.ma.sum((fcst < threshold) & (obs >= threshold))   # Miss
            d = np.ma.sum((fcst < threshold) & (obs < threshold))    # CR
            if(a + c > 0 and b + d > 0):
               y[i] = a / 1.0 / (a + c)
               x[i] = b / 1.0 / (b + d)
         # Add end points at 0,0 and 1,1:
         xx = x
         yy = y
         x = np.zeros([len(quantiles) + 2], 'float')
         y = np.zeros([len(quantiles) + 2], 'float')
         x[1:-1] = xx
         y[1:-1] = yy
         x[0] = 0
         y[0] = 0
         x[len(x) - 1] = 1
         y[len(y) - 1] = 1
         mpl.plot(x, y, style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
         if(self._labelQuantiles):
            for i in range(0, len(quantiles)):
               mpl.text(x[i + 1], y[i + 1], " %g%%" % quantiles[i], verticalalignment='center')
      mpl.plot([0, 1], [0, 1], color="k")
      mpl.axis([0, 1, 0, 1])
      mpl.xlabel("False alarm rate")
      mpl.ylabel("Hit rate")
      self._plot_perfect_score([0, 0, 1], [0, 1, 1])
      units = " " + data.get_units()
      mpl.title("Threshold: " + str(threshold) + units)
      mpl.grid()


# doClassic: Use the classic definition, by not varying the forecast threshold
#            i.e. using the same threshold for observation and forecast.
class DRoc(Output):
   _description = "Plots the receiver operating characteristics curve for "\
         "the deterministic forecast for a single threshold. Uses different "\
         "forecast thresholds to create points."
   _supX = False
   _reqThreshold = True

   def __init__(self, fthresholds=None, doNorm=False, doClassic=False):
      Output.__init__(self)
      self._doNorm = doNorm
      self._fthresholds = fthresholds
      self._doClassic = doClassic
      self._showThresholds = False

   def _plot_core(self, data):
      threshold = self._thresholds[0]   # Observation threshold
      if(threshold is None):
         verif.util.error("DRoc plot needs a threshold (use -r)")

      if(self._doClassic):
         fthresholds = [threshold]
      else:
         if(self._fthresholds is not None):
            fthresholds = self._fthresholds
         else:
            if(data.get_variable().name() == "Precip"):
               fthresholds = [0, 1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.005,
                     0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 100]
            else:
               N = 31
               fthresholds = np.linspace(threshold - 10, threshold + 10, N)

      F = data.get_num_files()
      labels = data.get_legend()
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F)
         data.set_axis("none")
         data.set_index(0)
         data.set_file_index(f)
         [obs, fcst] = data.get_scores(["obs", "fcst"])

         y = np.nan * np.zeros([len(fthresholds), 1], 'float')
         x = np.nan * np.zeros([len(fthresholds), 1], 'float')
         for i in range(0, len(fthresholds)):
            fthreshold = fthresholds[i]
            x[i] = verif.metric.Fa().compute_obs_fcst(obs, fcst + threshold - fthresholds[i], [threshold, np.inf])
            y[i] = verif.metric.Hit().compute_obs_fcst(obs, fcst + threshold - fthresholds[i], [threshold, np.inf])
            if(self._showThresholds and (not np.isnan(x[i]) and
                  not np.isnan(y[i]) and f == 0)):
               mpl.text(x[i], y[i], "%2.1f" % fthreshold, color=color)
         if(not self._doNorm):
            # Add end points at 0,0 and 1,1:
            xx = x
            yy = y
            x = np.zeros([len(fthresholds) + 2, 1], 'float')
            y = np.zeros([len(fthresholds) + 2, 1], 'float')
            x[1:-1] = xx
            y[1:-1] = yy
            x[0] = 1
            y[0] = 1
            x[len(x) - 1] = 0
            y[len(y) - 1] = 0
         mpl.plot(x, y, style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
      if(self._doNorm):
         xlim = mpl.xlim()
         ylim = mpl.ylim()
         q0 = max(abs(xlim[0]), abs(ylim[0]))
         q1 = max(abs(xlim[1]), abs(ylim[1]))
         mpl.plot([-q0, q1], [-q0, q1], 'k--')
         mpl.xlabel("Normalized false alarm rate")
         mpl.ylabel("Normalized hit rate")
      else:
         mpl.plot([0, 1], [0, 1], color="k")
         mpl.axis([0, 1, 0, 1])
         mpl.xlabel("False alarm rate")
         mpl.ylabel("Hit rate")
         self._plot_perfect_score([0, 0, 1], [0, 1, 1])
      units = " " + data.get_units()
      mpl.title("Threshold: " + str(threshold) + units)
      mpl.grid()


class DRocNorm(DRoc):
   _description = "Same as DRoc, except the hit and false alarm rates are transformed using the " \
            "inverse of the standard normal distribution in order to highlight the extreme " \
            "values."

   def __init__(self):
      DRoc.__init__(self, doNorm=True)


class DRoc0(DRoc):
   _description = "Same as DRoc, except don't use different forecast thresholds: Use the "\
      "same\n threshold for forecast and obs."

   def __init__(self):
      DRoc.__init__(self, doNorm=False, doClassic=True)


class Against(Output):
   _description = "Plots the forecasts for each pair of configurations against each other. "\
   "Colours indicate which configuration had the best forecast (but only if the difference is "\
   "more than 10% of the standard deviation of the observation)."
   _default_axis = "none"
   _supThreshold = False
   _supX = False
   # How big difference should colour kick in (in number of STDs)?
   _minStdDiff = 0.1

   def _plot_core(self, data):
      F = data.get_num_files()
      if(F < 2):
         verif.util.error("Cannot use Against plot with less than 2 configurations")

      data.set_axis("none")
      data.set_index(0)
      labels = data.get_legend()
      for f0 in range(0, F):
         for f1 in range(0, F):
            if(f0 != f1 and (F != 2 or f0 == 0)):
               if(F > 2):
                  mpl.subplot(F, F, f0 + f1 * F + 1)
               data.set_file_index(f0)
               x = data.get_scores("fcst")[0].flatten()
               data.set_file_index(f1)
               y = data.get_scores("fcst")[0].flatten()
               lower = min(min(x), min(y))
               upper = max(max(x), max(y))

               mpl.plot(x, y, "x", mec="k", ms=self._ms / 2, mfc="k",
                     zorder=-1000)

               # Show which forecast is better
               data.set_file_index(f0)
               [obsx, x] = data.get_scores(["obs", "fcst"])
               data.set_file_index(f1)
               [obsy, y] = data.get_scores(["obs", "fcst"])
               x = x.flatten()
               y = y.flatten()
               obs = obsx.flatten()

               mpl.plot(x, y, "s", mec="k", ms=self._ms / 2, mfc="w",
                     zorder=-500)

               std = np.std(obs) / 2
               minDiff = self._minStdDiff * std
               if(len(x) == len(y)):
                  N = 5
                  for k in range(0, N):
                     Ix = abs(obs - y) > abs(obs - x) + std * k / N
                     Iy = abs(obs - y) + std * k / N < abs(obs - x)
                     alpha = k / 1.0 / N
                     mpl.plot(x[Ix], y[Ix], "r.", ms=self._ms, alpha=alpha)
                     mpl.plot(x[Iy], y[Iy], "b.", ms=self._ms, alpha=alpha)

               # Contour of the frequency
               # q = np.histogram2d(x[1,:], x[0,:], [np.linspace(lower,upper,100), np.linspace(lower,upper,100)])
               # [X,Y] = np.meshgrid(q[1],q[2])
               # mpl.contour(X[1:,1:],Y[1:,1:],q[0],[1,100],zorder=90)

               mpl.xlabel(labels[f0], color="r")
               mpl.ylabel(labels[f1], color="b")
               mpl.grid()
               xlim = mpl.xlim()
               ylim = mpl.ylim()
               lower = min(xlim[0], ylim[0])
               upper = max(xlim[1], ylim[1])
               mpl.xlim([lower, upper])
               mpl.ylim([lower, upper])
               mpl.plot([lower, upper], [lower, upper], '--',
                        color=[0.3, 0.3, 0.3], lw=3, zorder=100)
               if(F == 2):
                  break
      mpl.gca().set_aspect(1)

   def _legend(self, data, names=None):
      pass


class Taylor(Output):
   _description = "Taylor diagram showing correlation and forecast standard deviation. Use '-x none' to collapse all data showing only one point.  Otherwise, the whole graph is normalized by the standard deviation of the observations."
   _supThreshold = True
   _supX = True
   _legLoc = "upper left"

   def _plot_core(self, data):
      data.set_axis(self._xaxis)
      data.set_index(0)
      labels = data.get_legend()
      F = data.get_num_files()

      # Plot points
      maxstd = 0
      for f in range(0, F):
         data.set_file_index(f)
         color = self._get_color(f, F)
         style = self._get_style(f, F)

         size = data.get_axis_size()
         corr = np.zeros(size, 'float')
         std = np.zeros(size, 'float')
         stdobs = np.zeros(size, 'float')
         for i in range(0, size):
            data.set_index(i)
            [obs, fcst] = data.get_scores(["obs", "fcst"])
            if(len(obs) > 0 and len(fcst) > 0):
               corr[i] = np.corrcoef(obs, fcst)[1, 0]
               std[i] = np.sqrt(np.var(fcst))
               stdobs[i] = np.sqrt(np.var(obs))
         # Normalize
         if size > 1:
            std = std / stdobs
            stdobs = 1.0
            xlabel = "Normalized standard deviation"
            crmseLabel = "Norm CRMSE"
            minCrmseLabel = "Min norm CRMSE"
         else:
            stdobs = verif.util.nanmean(stdobs)
            xlabel = "Standard deviation (" + data.get_units() + ")"
            crmseLabel = "CRMSE"
            minCrmseLabel = "Min CRMSE"

         maxstd = max(maxstd, max(std))
         ang = np.arccos(corr)
         x = std * np.cos(ang)
         y = std * np.sin(ang)
         mpl.plot(x, y, style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)

      # Set axis limits
      # Enforce a minimum radius beyond the obs-radius
      if(maxstd < 1.25 * stdobs):
         maxstd = 1.25 * stdobs
      maxstd = int(np.ceil(maxstd))
      # Allow for some padding outside the outer ring
      mpl.xlim([-maxstd * 1.05, maxstd * 1.05])
      mpl.ylim([0, maxstd * 1.05])
      xticks = mpl.xticks()[0]
      mpl.xticks(xticks[xticks >= 0])
      mpl.xlim([-maxstd * 1.05, maxstd * 1.05])
      mpl.ylim([0, maxstd * 1.05])
      mpl.text(np.sin(np.pi / 4) * maxstd, np.cos(np.pi / 4) * maxstd,
            "Correlation", rotation=-45, fontsize=self._labfs,
            horizontalalignment="center", verticalalignment="bottom")
      mpl.gca().yaxis.set_visible(False)
      # Remove box around plot
      mpl.gca().spines['bottom'].set_visible(False)
      mpl.gca().spines['top'].set_visible(False)
      mpl.gca().spines['left'].set_visible(False)
      mpl.gca().spines['right'].set_visible(False)
      mpl.gca().xaxis.set_ticks_position('bottom')
      mpl.xlabel(xlabel)

      # Draw obs point/lines
      orange = [1, 0.8, 0.4]
      self._draw_circle(stdobs, style='-', lw=5, color=orange)
      mpl.plot(stdobs, 0, 's-', color=orange, label="Obs", mew=2, ms=self._ms, clip_on=False)

      # Draw diagonals
      corrs = [-1, -0.99, -0.95, -0.9, -0.8, -0.5, 0, 0.5, 0.8, 0.9, 0.95,
            0.99]
      for i in range(0, len(corrs)):
         ang = np.arccos(corrs[i])  # Mathematical angle
         x = np.cos(ang) * maxstd
         y = np.sin(ang) * maxstd
         mpl.plot([0, x], [0, y], 'k--')
         mpl.text(x, y, str(corrs[i]), verticalalignment="bottom", fontsize=self._labfs)

      # Draw CRMSE rings
      xticks = mpl.xticks()[0]
      self._draw_circle(0, style="-", color="gray", lw=3, label=crmseLabel)
      Rs = np.linspace(0, 2 * max(xticks), 4 * max(xticks) / (xticks[1] -
         xticks[0]) + 1)
      for R in Rs:
         if(R > 0):
            self._draw_circle(R, xcenter=stdobs, ycenter=0, maxradius=maxstd, style="-", color="gray", lw=3)
            x = np.sin(-np.pi / 4) * R + stdobs
            y = np.cos(np.pi / 4) * R
            if(x ** 2 + y ** 2 < maxstd ** 2):
               mpl.text(x, y, str(R), horizontalalignment="right",
                     verticalalignment="bottom", fontsize=self._labfs,
                     color="gray")

      # Draw minimum CRMSE
      self._draw_circle(stdobs/2, xcenter=stdobs/2, ycenter=0, style="--",
            color="orange", lw=3, label=minCrmseLabel, zorder=0)

      # Draw std rings
      for X in mpl.xticks()[0]:
         if(X <= maxstd):
            self._draw_circle(X, style=":")
      self._draw_circle(maxstd, style="-", lw=3)

      # Draw bottom line
      mpl.plot([-maxstd, maxstd], [0, 0], "k-", lw=3)
      mpl.gca().set_aspect(1)


class Performance(Output):
   _description = "Categorical performance diagram showing POD, FAR, bias, and Threat score. Also shows the scores the forecasts would attain by using different forecast thresholds (turn off using -simple)"
   _supThreshold = True
   _reqThreshold = True
   _supX = True
   _legLoc = "upper left"
   _reference = "Roebber, P.J., 2009: Visualizing multiple measures of forecast quality. Wea. Forecasting, 24, 601-608."

   # Should lines be drawn to show how the scores can vary with chosen forecast
   # threshold?
   def _show_potential(self):
      return not self._simple

   def _plot_core(self, data):
      data.set_axis(self._xaxis)
      data.set_index(0)
      labels = data.get_legend()
      F = data.get_num_files()

      # Plot points
      maxstd = 0
      for f in range(0, F):
         data.set_file_index(f)
         color = self._get_color(f, F)
         style = self._get_style(f, F, False)

         size = data.get_axis_size()
         for t in range(0, len(self._thresholds)):
            threshold = self._thresholds[t]
            sr = np.zeros(size, 'float')
            pod = np.zeros(size, 'float')
            Far = verif.metric.Far()
            Hit = verif.metric.Hit()
            for i in range(0, size):
               data.set_index(i)
               [obs, fcst] = data.get_scores(["obs", "fcst"])
               fa = Far.compute_obs_fcst(obs, fcst, [threshold, np.inf])
               hit = Hit.compute_obs_fcst(obs, fcst, [threshold, np.inf])
               sr[i] = 1 - fa
               pod[i] = hit

               # Compute the potential that the forecast can attan by using
               # different forecast thresholds
               if self._show_potential():
                  J = 20
                  dx = threshold - np.percentile(np.unique(np.sort(fcst)), np.linspace(0, 100, J))
                  # Put a point in forecast point (so that the line goes
                  # through the point
                  dx = np.unique(np.sort(np.append(dx, 0)))
                  # Alternatively, nudge the closest point to 0
                  # Iclosest = np.argmin(np.abs(dx))
                  # dx[Iclosest] = 0

                  J = len(dx)
                  x = np.zeros(J, 'float')
                  y = np.zeros(J, 'float')
                  for j in range(0, J):
                     x[j] = 1 - Far.compute_obs_fcst(obs, fcst + dx[j], [threshold, np.inf])
                     y[j] = Hit.compute_obs_fcst(obs, fcst + dx[j], [threshold, np.inf])
                  mpl.plot(x, y, ".-", color=color, ms=3*self._lw, lw=2*self._lw, zorder=-100, alpha=0.3)

            label = ""
            if t == 0:
               label = labels[f]
            mpl.plot(sr, pod, style, color=color, label=label, lw=self._lw, ms=self._ms)

      # Plot bias lines
      biases = [0.3, 0.5, 0.8, 1, 1.3, 1.5, 2, 3, 5, 10]
      for i in range(0, len(biases)):
         bias = biases[i]
         label = ""
         if i == 0:
            label = "Bias frequency"
         mpl.plot([0, 1], [0, bias], 'k-', label=label)
         if(bias <= 1):
            mpl.text(1, bias, "%2.1f" % (bias))
         else:
            mpl.text(1.0/bias, 1, "%2.1f" % (bias))

      # Plot threat score lines
      threats = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      for i in range(0, len(threats)):
         threat = threats[i]
         x = np.linspace(threat, 1, 100)
         label = ""
         if i == 0:
            label = "Threat score"
         y = 1.0 / (1 + 1.0/threat - 1.0 / x)
         mpl.plot(x, y, 'k--', label=label)
         xx = 2.0 / (1 + 1.0/threat)
         mpl.text(xx, xx, str(threat))

      mpl.xlabel("Success ratio (1 - FAR)")
      mpl.ylabel("Probability of detection")
      mpl.xlim([0, 1])
      mpl.ylim([0, 1])
      mpl.grid()
      mpl.gca().set_aspect(1)


class Error(Output):
   _description = "Decomposition of RMSE into systematic and unsystematic components"
   _supThreshold = False
   _supX = True
   _default_axis = "none"

   def _plot_core(self, data):
      data.set_axis(self._xaxis)
      data.set_index(0)
      labels = data.get_legend()
      F = data.get_num_files()

      mpl.gca().set_aspect(1)
      mpl.xlabel("Unsystematic error (CRMSE, " + data.get_units() + ")")
      mpl.ylabel("Systematic error (Bias, " + data.get_units() + ")")

      # Plot points
      size = data.get_axis_size()
      serr = np.nan * np.zeros([size, F], 'float')
      uerr = np.nan * np.zeros([size, F], 'float')
      rmse = np.nan * np.zeros([size, F], 'float')
      for f in range(0, F):
         data.set_file_index(f)
         color = self._get_color(f, F)
         style = self._get_style(f, F, connectingLine=False)

         for i in range(0, size):
            data.set_index(i)
            [obs, fcst] = data.get_scores(["obs", "fcst"])
            mfcst = np.mean(fcst)
            mobs = np.mean(obs)
            if(len(obs) > 0 and len(fcst) > 0):
               serr[i, f] = np.mean(obs - fcst)
               rmse[i, f] = np.sqrt(np.mean((obs - fcst) ** 2))
               uerr[i, f] = np.sqrt(rmse[i, f] ** 2 - serr[i, f] ** 2)
         mpl.plot(uerr[:, f], serr[:, f], style, color=color, label=labels[f],
               lw=self._lw, ms=self._ms)
      xlim = mpl.xlim()
      ylim = mpl.ylim()

      # Draw rings
      for f in range(0, F):
         color = self._get_color(f, F)
         style = self._get_style(f, F, lineOnly=True)
         self._draw_circle(verif.util.nanmean(rmse[:, f]), style=style, color=color)

      # Set axis limits
      maxx = xlim[1]
      maxy = ylim[1]
      miny = min(0, ylim[0])
      # Try to enforce the x-axis and y-axis to be roughly the same size
      if(maxy - miny < maxx / 2):
         maxy = maxx
      elif(maxy - miny > maxx * 2):
         maxx = maxy - miny
      mpl.xlim([0, maxx])  # Not possible to have negative CRMSE
      mpl.ylim([miny, maxy])

      # Draw standard RMSE rings
      for X in mpl.xticks()[0]:
         self._draw_circle(X, style=":")

      mpl.plot([0, maxx], [0, 0], 'k-', lw=2)  # Draw x-axis line
      mpl.grid()


class Marginal(Output):
   _description = "Show marginal distribution for different thresholds"
   _reqThreshold = True
   _supX = False
   _experimental = True

   def __init__(self):
      Output.__init__(self)

   def _plot_core(self, data):
      labels = data.get_legend()

      F = data.get_num_files()

      data.set_axis("none")
      data.set_index(0)
      data.set_file_index(0)
      clim = np.zeros(len(self._thresholds), 'float')
      for f in range(0, F):
         x = self._thresholds
         y = np.zeros([len(self._thresholds)], 'float')
         for t in range(0, len(self._thresholds)):
            threshold = self._thresholds[t]
            data.set_file_index(f)
            data.set_axis("none")
            data.set_index(0)
            var = data.get_p_var(threshold)
            [obs, p] = data.get_scores(["obs", var])

            color = self._get_color(f, F)
            style = self._get_style(f, F)

            if(self._binType == "below"):
               p = p
               obs = obs < threshold
            elif(self._binType == "above"):
               p = 1 - p
               obs = obs > threshold
            else:
               verif.util.error("Bin type must be one of 'below' or 'above' for reliability plot")

            clim[t] = np.mean(obs)
            y[t] = np.mean(p)

         label = labels[f]
         mpl.plot(x, y, style, color=color, lw=self._lw, ms=self._ms, label=label)
      self._plot_obs(x, clim)

      mpl.ylim([0, 1])
      mpl.xlabel(data.get_axis_label("threshold"))
      mpl.ylabel("Marginal probability")
      mpl.grid()


class Freq(Output):
   _description = "Show frequency of obs and forecasts"
   _reqThreshold = True
   _supX = False
   _experimental = True

   def __init__(self):
      Output.__init__(self)

   def _plot_core(self, data):
      labels = data.get_legend()

      F = data.get_num_files()

      data.set_axis("none")
      data.set_index(0)
      data.set_file_index(0)

      for f in range(0, F):
         # Setup x and y: When -b within, we need one less value in the array
         N = len(self._thresholds)
         x = self._thresholds
         if(self._binType == "within"):
            N = len(self._thresholds) - 1
            x = (self._thresholds[1:] + self._thresholds[:-1]) / 2
         y = np.zeros(N, 'float')
         clim = np.zeros(N, 'float')
         for t in range(0, N):
            threshold = self._thresholds[t]
            data.set_file_index(f)
            data.set_axis("none")
            data.set_index(0)
            [obs, fcst] = data.get_scores(["obs", "fcst"])

            color = self._get_color(f, F)
            style = self._get_style(f, F)

            if(self._binType == "below"):
               fcst = fcst < threshold
               obs = obs < threshold
            elif(self._binType == "above"):
               fcst = fcst > threshold
               obs = obs > threshold
            elif(self._binType == "within"):
               fcst = (fcst >= threshold) & (fcst < self._thresholds[t + 1])
               obs = (obs >= threshold) & (obs < self._thresholds[t + 1])

            clim[t] = np.mean(obs)
            y[t] = np.mean(fcst)

         label = labels[f]
         mpl.plot(x, y, style, color=color, lw=self._lw, ms=self._ms, label=label)
      self._plot_obs(x, clim)

      mpl.ylim([0, 1])
      mpl.xlabel(data.get_axis_label("threshold"))
      mpl.ylabel("Frequency " + self._binType)
      mpl.grid()


class InvReliability(Output):
   _description = "Reliability diagram for a certain quantile (-r)"
   _reqThreshold = True
   _supX = False
   _experimental = True

   def __init__(self):
      Output.__init__(self)

   def _show_count(self):
      return False

   def _plot_core(self, data):
      labels = data.get_legend()

      F = data.get_num_files()
      ax = mpl.gca()
      quantiles = self._thresholds
      if(self._show_count()):
         if(quantiles[0] < 0.5):
            axi = mpl.axes([0.66, 0.65, 0.2, 0.2])
         else:
            axi = mpl.axes([0.66, 0.15, 0.2, 0.2])
      mpl.sca(ax)

      data.set_axis("none")
      data.set_index(0)
      data.set_file_index(0)
      for t in range(0, len(quantiles)):
         quantile = self._thresholds[t]
         var = data.get_q_var(quantile)
         [obs, p] = data.get_scores(["obs", var])

         # Determine the number of bins to use # (at least 11, at most 25)
         N = min(25, max(11, int(len(obs) / 1000)))
         N = 21
         edges = np.linspace(0, 20, N + 1)
         if(data.get_variable().name() == "Precip"):
            edges = np.linspace(0, np.sqrt(verif.util.nanmax(obs)), N + 1) ** 2
         else:
            edges = np.linspace(verif.util.nanmin(obs), verif.util.nanmax(obs), N + 1)

         x = np.zeros([len(edges) - 1, F], 'float')
         y = np.nan * np.zeros([F, len(edges) - 1], 'float')
         n = np.zeros([F, len(edges) - 1], 'float')
         v = np.zeros([F, len(edges) - 1], 'float')
         # Draw reliability lines
         for f in range(0, F):
            color = self._get_color(f, F)
            style = self._get_style(f, F)
            data.set_file_index(f)
            data.set_axis("none")
            data.set_index(0)
            var = data.get_q_var(quantile)
            [obs, p] = data.get_scores(["obs", var])

            obs = obs <= p

            # Compute frequencies
            for i in range(0, len(edges) - 1):
               q = (p >= edges[i]) & (p < edges[i + 1])
               I = np.where(q)[0]
               if(len(I) > 0):
                  n[f, i] = len(obs[I])
                  # Need at least 10 data points to be valid
                  if(n[f, i] >= 2):
                     y[f, i] = np.mean(obs[I])
                     v[f, i] = np.var(obs[I])
                  x[i, f] = np.mean(p[I])

            label = labels[f]
            if(not t == 0):
               label = ""
            mpl.plot(x[:, f], y[f], style, color=color, lw=self._lw,
                  ms=self._ms, label=label)
         self._plot_obs(edges, 0 * edges + quantile, label="")

         # Draw confidence bands (do this separately so that these lines don't
         # sneak into the legend)
         for f in range(0, F):
            color = self._get_color(f, F)
            self._plot_confidence(x[:, f], y[f], v[f], n[f], color=color)
            if(self._show_count()):
               axi.plot(x[:, f], n[f], style, color=color, lw=self._lw, ms=self._ms)
               axi.xaxis.set_major_locator(mpl.NullLocator())
               axi.set_yscale('log')
               axi.set_title("Number")
      mpl.sca(ax)
      mpl.ylim([0, 1])
      color = "gray"
      mpl.xlabel(data.get_variable_and_units())
      mpl.ylabel("Observed frequency")
      units = " " + data.get_units()
      mpl.title("Quantile: " + str(quantile * 100) + "%")
