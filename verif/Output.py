# -*- coding: ISO-8859-1 -*-
import matplotlib.pyplot as mpl
import re
import datetime
import verif.Common as Common
import verif.Metric as Metric
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')
#from matplotlib.dates import *
import os
import inspect

def getAllOutputs():
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return temp

def isNumber(s): # tchui (25/05/15)
   try:
      float(s)
      return True
   except ValueError:
      return False

class Output:
   _description  = ""
   _defaultAxis = "offset"
   _defaultBinType = "above"
   _reqThreshold = False
   _supThreshold = True
   _supX = True
   _experimental = False
   _legLoc = "best" # Where should the legend go?

   def __init__(self):
      self._filename = None
      self._thresholds = [None]
      leg = None
      self.default_lines = ['-','-','-','--']
      self.default_markers = ['o', '', '.', '']
      self.default_colors = ['r',  'b', 'g', [1,0.73,0.2], 'k']
      self._lc = None
      self._ls = None
      self.colors = None
      self.styles = None
      self._ms = 8
      self._lw = 2
      self._labfs = 16
      self._tickfs = 16
      self._legfs = 16
      self._figsize = [5,8]
      self._showMargin = True
      self._xrot = 0
      self._minlth = None
      self._majlth = None
      self._majwid = None
      self._bot = None #######
      self._top = None #######
      self._left = None #######
      self._right = None #######
      #self._pad = pad ######
      self._xaxis = self.defaultAxis()
      self._binType = self.defaultBinType()
      self._showPerfect = False
      self._dpi = 100
      self._xlim = None
      self._ylim = None
      self._clim = None
      self._title = None
      self._xlabel = None
      self._ylabel = None
      self._xticks = None
      self._yticks = None
      self._tight = False

   @classmethod
   def defaultAxis(cls):
      return cls._defaultAxis

   @classmethod
   def defaultBinType(cls):
      return cls._defaultBinType

   @classmethod
   def requiresThresholds(cls):
      return cls._reqThreshold

   @classmethod
   def supportsX(cls):
      return cls._supX

   @classmethod
   def supportsThreshold(cls):
      return cls._supThreshold

   @classmethod
   def description(cls):
      extra = ""
      if(cls._experimental):
         extra = " " + Common.experimental()
      return cls._description + extra

   @classmethod
   def summary(cls):
      return cls.description()

   # Produce output independently for each value along this axis
   def setAxis(self, axis):
      if(axis != None):
         self._xaxis = axis

   def setBinType(self, binType):
      if(binType != None):
         self._binType = binType

   def setThresholds(self, thresholds):
      if(thresholds == None):
         thresholds = [None]
      thresholds = np.array(thresholds)
      self._thresholds = thresholds
   def setFigsize(self, size):
      self._figsize = size
   def setFilename(self, filename):
      self._filename = filename
   def setLegend(self, legend):
      self._legNames = legend
   def setLegLoc(self, legLoc): # tchui (25/05/15)
      self._legLoc = legLoc
   def setShowMargin(self, showMargin):
      self._showMargin = showMargin
   def setDpi(self, dpi):
      self._dpi = dpi
   def setXLim(self, lim):
      if(len(lim) != 2):
         Common.error("xlim must be a vector of length 2")
      self._xlim = lim
   def setYLim(self, lim):
      if(len(lim) != 2):
         Common.error("ylim must be a vector of length 2")
      self._ylim = lim
   def setCLim(self, lim):
      if(len(lim) != 2):
         Common.error("clim must be a vector of length 2")
      self._clim = lim
   def setMarkerSize(self, ms):
      self._ms = ms
   def setLineWidth(self, lw):
      self._lw = lw
   def setLineColor(self,lc):
      self._lc = lc
   def setLineStyle(self,ls):
      self._ls = ls
   def setTickFontSize(self, fs):
      self._tickfs = fs
   def setLabFontSize(self, fs):
      self._labfs = fs
   def setLegFontSize(self, fs):
      self._legfs = fs
   def setXRotation(self, xrot):
      self._xrot = xrot
   def setMinorLength(self, minlth):
      self._minlth = minlth
   def setMajorLength(self, majlth):
      self._majlth = majlth
   def setMajorWidth(self, majwid):
      self._majwid = majwid
   def setBottom(self, bot):
      self._bot = bot
   def setTop(self, top):
      self._top = top
   def setLeft(self, left):
      self._left = left
   def setRight(self, right):
      self._right = right
   #def setPad(self, pad):
   #   self._pad = pad
   def setShowPerfect(self, showPerfect):
      self._showPerfect = showPerfect
   def setYlabel(self, ylabel):
      self._ylabel = ylabel
   def setXlabel(self, xlabel):
      self._xlabel = xlabel
   def setTitle(self, title):
      self._title = title
   def setXticks(self, xticks):
      self._xticks = xticks
   def setYticks(self, yticks):
      self._yticks = yticks
   def setTight(self,tight): #potato
      self._tight = tight


   # Public
   # Call this to create a plot, saves to file
   def plot(self, data):
      self._plotCore(data)
      self._adjustAxes()
      self._legend(data, self._legNames)
      self._savePlot(data)
   # Call this to write text output
   def text(self, data):
      self._textCore(data)
   # Draws a map of the data
   def map(self, data):
      self._mapCore(data)
      #self._legend(data, self._legNames)
      self._savePlot(data)

   def _getLegendNames(self, data):
      if(self._legNames != None):
         names = self._legNames
      else:
         names = data.getShortNames()
      return(names)

   def _plotPerfectScore(self, x, perfect, color="gray", zorder=-1000):
      if(perfect == None):
         return
      if(self._showPerfect):
         # Make 'perfect' same length as 'x'
         if(not hasattr(perfect, "__len__")):
            perfect = perfect*np.ones(len(x), 'float')
         mpl.plot(x, perfect, '-', lw=7, color=color, label="ideal", zorder=zorder)

   # Implement these methods
   def _plotCore(self, data):
      Common.error("This type does not plot")
   def _textCore(self, data):
      Common.error("This type does not output text")
   def _mapCore(self, data):
      Common.error("This type does not produce maps")

   # Helper functions
   def _getColor(self, i, total):
      if(self._lc != None):
         firstList = self._lc.split(",")
         numList = []
         finalList = []

         for string in firstList:
            if("[" in string):   # for rgba args
               if(not numList):
                  string = string.replace("[","")
                  numList.append(float(string))
               else:
                  Common.error("Invalid rgba arg \"{}\"".format(string))

            elif("]" in string):
               if(numList):
                  string = string.replace("]","")
                  numList.append(float(string))
                  finalList.append(numList)
                  numList = []
               else:
                  Common.error("Invalid rgba arg \"{}\"".format(string))

            elif(isNumber(string)): # append to rgba lists if present, otherwise grayscale intensity
               if(numList):
                  numList.append(float(string))
               else:
                  finalList.append(string)

            else:
               if(not numList): # string args and hexcodes
                  finalList.append(string)
               else:
                  Common.error("Cannot read color args.")
         self.colors = finalList
         return self.colors[i % len(self.colors)]

      else: # use default colours if no colour input given
         self.colors = self.default_colors
         return self.colors[i % len(self.default_colors)]


   def _getStyle(self, i, total, connectingLine=True, lineOnly=False): # edited by tchui (25/05/15)
      if(self._ls != None):
         listStyles = self._ls.split(",")
         I = i % len(listStyles) # loop through input linestyles (independent of colors)
         return listStyles[I]

      else: # default linestyles
         I = (i / len(self.colors)) % len(self.default_lines)
         line   = self.default_lines[I]
         marker = self.default_markers[I]
         if(lineOnly):
            return line
         if(connectingLine):
            return line + marker
         return marker


   # Saves to file, set figure size
   def _savePlot(self, data):
      if(self._figsize != None):
         mpl.gcf().set_size_inches(int(self._figsize[0]), int(self._figsize[1]))
      if(not self._showMargin):
         Common.removeMargin()
      if(self._filename != None):
         mpl.savefig(self._filename, bbox_inches='tight', dpi=self._dpi)
      else:
         fig = mpl.gcf()
         fig.canvas.set_window_title(data.getFilenames()[0])
         mpl.show()
   def _legend(self, data, names=None):
      if(names == None):
         mpl.legend(loc=self._legLoc,prop={'size':self._legfs})
      else:
         mpl.legend(names, loc=self._legLoc,prop={'size':self._legfs})
   def _getThresholdLimits(self, thresholds):
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
         x = [(lowerT[i] + upperT[i])/2 for i in range(0, len(lowerT))]
      else:
         Common.error("Unrecognized bintype")
      return [lowerT,upperT,x]

   def _setYAxisLimits(self, metric):
      currYlim = mpl.ylim()
      ylim = [metric.min(), metric.max()]
      if(ylim[0] == None):
         ylim[0] = currYlim[0]
      if(ylim[1] == None):
         ylim[1] = currYlim[1]
      mpl.ylim(ylim)

   def _adjustAxes(self):
      # Apply adjustements to all subplots
      for ax in mpl.gcf().get_axes():
         # Tick font sizes
         for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(self._tickfs)
         for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(self._tickfs)
         ax.set_xlabel(ax.get_xlabel(), fontsize=self._labfs)
         ax.set_ylabel(ax.get_ylabel(), fontsize=self._labfs)
         #mpl.rcParams['axes.labelsize'] = self._labfs

         # Tick lines
         if(len(mpl.yticks()[0]) >= 2 and len(mpl.xticks()[0]) >= 2):
            # matplotlib crashes if there are fewer than 2 tick lines
            # when determining where to put minor ticks
            mpl.minorticks_on()
         if(not self._minlth == None):
            mpl.tick_params('both', length=self._minlth, which='minor')
         if(not self._majlth == None):
            mpl.tick_params('both', length=self._majlth, width=self._majwid, which='major')
         for label in ax.get_xticklabels():
            label.set_rotation(self._xrot)

      for ax in mpl.gcf().get_axes():
         if(self._xlim != None):
            mpl.xlim(self._xlim)
         if(self._ylim != None):
            mpl.ylim(self._ylim)
         if(self._clim != None):
            mpl.clim(self._clim)

      # Labels
      if(self._xlabel != None):
         mpl.xlabel(self._xlabel)
      if(self._ylabel != None):
         mpl.ylabel(self._ylabel)
      if(self._title != None):
         mpl.title(self._title)

      # Ticks
      if(self._xticks != None):
         if(len(self._xticks) <= 1):
            Common.error("Xticks must have at least 2 values")
         mpl.xticks(self._xticks)
      if(self._yticks != None):
         if(len(self._yticks) <= 1):
            Common.error("Yticks must have at least 2 values")
         mpl.yticks(self._yticks)

      # Margins
      mpl.gcf().subplots_adjust(bottom=self._bot, top=self._top, left=self._left, right=self._right)

   def _plotObs(self, x, y, isCont=True, zorder=0):
      if(isCont):
         mpl.plot(x, y,  ".-", color="gray", lw=5, label="obs", zorder=zorder)
      else:
         mpl.plot(x, y,  "o", color="gray", ms=self._ms, label="obs", zorder=zorder)

   # maxradius: Don't let the circle go outside an envelope circle with this radius (centered on the origin)
   def _drawCircle(self, radius, xcenter=0, ycenter=0, maxradius=np.inf, style="--", color="k", lw=1, label=""):
      angles = np.linspace(-np.pi/2, np.pi/2, 360)
      x = np.sin(angles)*radius + xcenter
      y = np.cos(angles)*radius + ycenter

      # Only keep points within the circle
      I = np.where(x**2+y**2 < maxradius**2)[0]
      if(len(I) == 0):
         return
      x = x[I]
      y = y[I]
      mpl.plot(x,  y,style,color=color,lw=lw, zorder=-100, label=label)
      mpl.plot(x, -y,style,color=color,lw=lw, zorder=-100)

   def _plotConfidence(self, x, y, variance, n, color):
      #variance = y*(1-y) # For bins

      # Remove missing points
      I = np.where(n != 0)[0]
      if(len(I) == 0):
         return
      x = x[I]
      y = y[I]
      variance = variance[I]
      n = n[I]

      z = 1.96 # 95% confidence interval
      type = "wilson"
      style = "--"
      if type == "normal":
         mean = y
         lower = mean - z*np.sqrt(variance/n)
         upper = mean + z*np.sqrt(variance/n)
      elif type == "wilson":
         mean =  1/(1+1.0/n*z**2) * ( y + 0.5*z**2/n)
         upper = mean + 1/(1+1.0/n*z**2)*z*np.sqrt(variance/n + 0.25*z**2/n**2)
         lower = mean - 1/(1+1.0/n*z**2)*z*np.sqrt(variance/n + 0.25*z**2/n**2)
      mpl.plot(x, upper, style, color=color, lw=self._lw, ms=self._ms,label="")
      mpl.plot(x, lower, style, color=color, lw=self._lw, ms=self._ms,label="")
      Common.fill(x, lower, upper, color, alpha=0.3)

class Default(Output):
   _legLoc = "upper left"
   def __init__(self, metric):
      Output.__init__(self)
      # offsets, dates, location, locationElev, threshold
      self._metric = metric
      if(metric.defaultAxis() != None):
         self._xaxis = metric.defaultAxis()
      if(metric.defaultBinType() != None):
         self._binType = metric.defaultBinType()
      self._showRank = False
      self._showAcc  = False
      self._setLegSort = False

      # Settings
      self._mapLowerPerc = 0    # Lower percentile (%) to show in colourmap
      self._mapUpperPerc = 100  # Upper percentile (%) to show in colourmap
      self._mapLabelLocations = False # Show locationIds in map?

   def setShowRank(self, showRank):
      self._showRank = showRank

   def setLegSort(self,dls):
      self._setLegSort = dls

   def setShowAcc(self, showAcc):
      self._showAcc = showAcc

   def getXY(self, data):
      thresholds = self._thresholds
      axis = data.getAxis()

      [lowerT,upperT,xx] = self._getThresholdLimits(thresholds)
      if(axis != "threshold"):
         xx = data.getAxisValues()

      filenames = data.getFilenames()
      F = data.getNumFiles()
      y = None
      x = None
      for f in range(0, F):
         data.setFileIndex(f)
         yy = np.zeros(len(xx), 'float')
         if(axis == "threshold"):
            for i in range(0, len(lowerT)):
               yy[i] = self._metric.compute(data, [lowerT[i], upperT[i]])
         else:
            for i in range(0, len(lowerT)):
               yy = yy + self._metric.compute(data, [lowerT[i], upperT[i]])
            yy = yy / len(thresholds)

         if(sum(np.isnan(yy)) == len(yy)):
            Common.warning("No valid scores for " + filenames[f])
         if(y == None):
            y = np.zeros([F, len(yy)],'float')
            x = np.zeros([F, len(xx)],'float')
         y[f,:] = yy
         x[f,:] = xx
         if(self._showAcc):
            y[f,:] = np.nan_to_num(y[f,:])
            y[f,:] = np.cumsum(y[f,:])
      return [x,y]

   def _legend(self, data, names=None):
      mpl.legend(loc=self._legLoc,prop={'size':self._legfs})

   def _plotCore(self, data):

      data.setAxis(self._xaxis)

      # We have to derive the legend list here, because we might want to specify
      # the order
      labels = np.array(data.getFilenames())
      if(self._legNames): # append legend names to file list
         try:
            labels[0:len(self._legNames)]=self._legNames
         except ValueError:
            Common.error("Too many legend names")

      self._legNames = labels

      F = data.getNumFiles()
      [x,y] = self.getXY(data)

      # Sort legend entries such that the appear in the same order as the y-values of
      # the lines
      if(self._setLegSort):
         if(not self._showAcc):
            averages = (Common.nanmean(y,axis=1)) # averaging for non-acc plots
            ids = averages.argsort()[::-1]

         else:
            ends = y[:,-1]  # take last points for acc plots
            ids = ends.argsort()[::-1]

         self._legNames = [self._legNames[i] for i in ids]

      else:
         ids = range(0,F)

      if(self._xaxis == "none"):
         w = 0.8
         x = np.linspace(1-w/2,len(y)-w/2,len(y))
         mpl.bar(x,y, color='w', lw=self._lw)
         mpl.xticks(range(1,len(y)+1), labels)
      else:
         for f in range(0, F):
            color = self._getColor(ids[f], F) # colors and styles to follow labels
            style = self._getStyle(ids[f], F, data.isAxisContinuous())
            alpha = (1 if(data.isAxisContinuous()) else 0.55)
            mpl.plot(x[ids[f]], y[ids[f]], style, color=color, label=self._legNames[f], lw=self._lw, ms=self._ms, alpha=alpha)

         mpl.xlabel(data.getAxisLabel())
         mpl.ylabel(self._metric.label(data))

         mpl.gca().xaxis.set_major_formatter(data.getAxisFormatter())
         perfectScore = self._metric.perfectScore()
         self._plotPerfectScore(x[0], perfectScore)

      mpl.grid()
      if(not self._showAcc):
         self._setYAxisLimits(self._metric)

      if(self._tight):
         oldTicks=mpl.gca().get_xticks()
         diff = oldTicks[1] - oldTicks[0] # keep auto tick interval
         tickRange = np.arange(round(np.min(x)),round(np.max(x))+diff,diff)
         mpl.gca().set_xticks(tickRange) # make new ticks, to start from the first day of the desired interval
         mpl.autoscale(enable=True,axis=u'x',tight=True) # make xaxis tight

   def _textCore(self, data):
      thresholds = self._thresholds

      data.setAxis(self._xaxis)

      # Set configuration names
      names = self._getLegendNames(data)

      F     = data.getNumFiles()
      [x,y] = self.getXY(data)

      if(self._filename != None):
         sys.stdout = open(self._filename, 'w')

      maxlength = 0
      for name in names:
         maxlength = max(maxlength, len(name))
      maxlength = str(maxlength)

      # Header line
      fmt = "%-"+maxlength+"s"
      lineDesc = data.getAxisDescriptionHeader()
      lineDescN = len(lineDesc) + 2
      lineDescFmt = "%-" + str(lineDescN) + "s |"
      print lineDescFmt % lineDesc,
      if(data.getAxis() == "threshold"):
         descs = self._thresholds
      else:
         descs = data.getAxisDescriptions()
      for name in names:
         print fmt % name,
      print ""

      # Loop over rows
      for i in range(0, len(x[0])):
         print lineDescFmt % descs[i],
         self._printLine(y[:,i], maxlength, "float")

      # Print stats
      for func in [Common.nanmin, Common.nanmean, Common.nanmax, Common.nanstd]:
         name = func.__name__[3:]
         print lineDescFmt % name,
         values = np.zeros(F, 'float')
         for f in range(0,F):
            values[f] = func(y[f,:])
         self._printLine(values, maxlength, "float")

      # Print count stats
      for func in [Common.nanmin, Common.nanmax]:
         name = func.__name__[3:]
         print lineDescFmt % ("num " + name),
         values = np.zeros(F, 'float')
         for f in range(0,F):
            values[f] = np.sum(y[f,:] == func(y,axis=0))
         self._printLine(values, maxlength, "int")

   def _printLine(self, values , colWidth, type="float"):
      if(type == "int"):
         fmt  = "%-"+colWidth+"i"
      else:
         fmt     = "%-"+colWidth+".2f"
      missfmt = "%-"+colWidth+"s"
      minI    = np.argmin(values)
      maxI    = np.argmax(values)
      for f in range(0, len(values)):
         value = values[f]
         if(np.isnan(value)):
            txt = missfmt % "--"
         else:
            txt = fmt % value
         if(minI == f):
            print Common.green(txt),
         elif(maxI == f):
            print Common.red(txt),
         else:
            print txt,
      print ""

   def _mapCore(self, data):
      # Use the Basemap package if it is available
      # Note that the word 'map' is an object if Basemap is loaded
      # otherwise it is a shorthand name for matplotlib. This is possible
      # because Basemap shares the plotting command names with matplotlib
      hasBasemap = True
      try:
         from mpl_toolkits.basemap import Basemap
      except ImportError:
         Common.warning("Cannot load Basemap package")
         import matplotlib.pylab as map
         hasBasemap = False

      data.setAxis("location")
      labels = self._getLegendNames(data)
      F = data.getNumFiles()
      lats = data.getLats()
      lons = data.getLons()
      ids  = data.getLocationIds()
      dlat = (max(lats) - min(lats))
      dlon = (max(lons) - min(lons))
      llcrnrlat= max(-90, min(lats) - dlat/10)
      urcrnrlat= min(90, max(lats) + dlat/10)
      llcrnrlon= min(lons) - dlon/10
      urcrnrlon= max(lons) + dlon/10
      res = Common.getMapResolution(lats, lons)
      dx = pow(10,np.ceil(np.log10(max(lons) - min(lons))))/10
      dy = pow(10,np.ceil(np.log10(max(lats) - min(lats))))/10
      [x,y] = self.getXY(data)

      # Colorbar limits should be the same for all subplots
      clim = [Common.nanpercentile(y.flatten(), self._mapLowerPerc),
              Common.nanpercentile(y.flatten(), self._mapUpperPerc)]

      symmetricScore = False
      cmap=mpl.cm.jet
      if(clim[0] < 0 and clim[1] > 0):
         symmetricScore = True
         clim[0] = -max(-clim[0],clim[1])
         clim[1] = -clim[0]
         cmap=mpl.cm.RdBu

      # Forced limits
      if(self._clim != None):
         clim = self._clim

      std = Common.nanstd(y)
      minDiff = std/50

      for f in range(0, F):
         Common.subplot(f,F)
         if(hasBasemap):
            map = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,projection='mill', resolution=res)
            map.drawcoastlines(linewidth=0.25)
            map.drawcountries(linewidth=0.25)
            map.drawmapboundary()
            # map.drawparallels(np.arange(-90.,120.,dy),labels=[1,0,0,0])
            # map.drawmeridians(np.arange(0.,420.,dx),labels=[0,0,0,1])
            map.fillcontinents(color='coral',lake_color='aqua', zorder=-1)
            x0, y0 = map(lons, lats)
         else:
            x0 = lons
            y0 = lats
         I = np.where(np.isnan(y[f,:]))[0]
         map.plot(x0[I], y0[I], 'kx')

         isMax = (y[f,:] == np.amax(y,0)) & (y[f,:] > np.mean(y,0)+minDiff)
         isMin = (y[f,:] == np.amin(y,0)) & (y[f,:] < np.mean(y,0)-minDiff)
         isValid = (np.isnan(y[f,:])==0)
         if(self._showRank):
            lmissing = map.scatter(x0[I], y0[I], s=40, c="k", marker="x")
            lsimilar = map.scatter(x0[isValid], y0[isValid], s=40, c="w")
            lmax = map.scatter(x0[isMax], y0[isMax], s=40, c="r")
            lmin = map.scatter(x0[isMin], y0[isMin], s=40, c="b")
         else:
            s = 40
            map.scatter(x0, y0, c=y[f,:], s=s, cmap=cmap)#, linewidths = 1 + 2*isMax)
            cb = map.colorbar()
            cb.set_label(self._metric.label(data))
            cb.set_clim(clim)
            mpl.clim(clim)
         if(self._mapLabelLocations):
            for i in range(0,len(x0)):
               #mpl.text(x0[i], y0[i], "(%d,%d)" % (i,locs[i]))
               value = y[f,i]
               #if(value == max(y[:,i])):
               #   mpl.plot(x0[i], y0[i], 'ko', mfc=None, mec="k", ms=10)

               if(not np.isnan(value)):
                  #if(isMax[i]):
                  #   mpl.plot(x0[i], y0[i], 'w.', ms=30, alpha=0.2)
                  mpl.text(x0[i], y0[i], "%d %3.2f" % (ids[i],value))
         if(self._legNames != None):
            names = self._legNames
         else:
            names = data.getFilenames()
         mpl.title(names[f])

      # Legend
      if(self._showRank):
         lines = [lmin,lsimilar,lmax,lmissing]
         names = ["min", "similar", "max", "missing"]
         mpl.figlegend(lines, names, "lower center", ncol=4)

class Hist(Output):
   _reqThreshold = True
   _supThreshold = False
   def __init__(self, name):
      Output.__init__(self)
      self._name = name

      # Settings
      self._showPercent = True
   def getXY(self, data):
      F = data.getNumFiles()
      allValues = [0]*F
      edges = self._thresholds
      for f in range(0, F):
         data.setFileIndex(f)
         allValues[f] = data.getScores(self._name)

      xx = (edges[0:-1]+edges[1:])/2
      y = np.zeros([F, len(xx)],'float')
      x = np.zeros([F, len(xx)],'float')
      for f in range(0, F):
         data.setFileIndex(f)
         N = len(allValues[f][0])

         for i in range(0, len(xx)):
            if(i == len(xx)-1):
               I = np.where((allValues[f][0] >= edges[i]) & (allValues[f][0] <= edges[i+1]))[0]
            else:
               I = np.where((allValues[f][0] >= edges[i]) & (allValues[f][0] < edges[i+1]))[0]
            y[f,i] = len(I)*1.0#/N
         x[f,:] = xx
      return [x,y]

   def _plotCore(self, data):
      data.setAxis("none")
      labels = self._getLegendNames(data)
      F = data.getNumFiles()
      [x,y] = self.getXY(data)
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F)
         if(self._showPercent):
            y[f]= y[f]* 1.0 / sum(y[f]) * 100
         mpl.plot(x[f], y[f], style, color=color, label=labels[f], lw=self._lw, ms=self._ms)
      mpl.xlabel(data.getAxisLabel("threshold"))
      if(self._showPercent):
         mpl.ylabel("Frequency (%)")
      else:
         mpl.ylabel("Frequency")
      mpl.grid()

   def _textCore(self, data):
      data.setAxis("none")
      labels = self._getLegendNames(data)

      F     = data.getNumFiles()
      [x,y] = self.getXY(data)

      if(self._filename != None):
         sys.stdout = open(self._filename, 'w')

      maxlength = 0
      for label in labels:
         maxlength = max(maxlength, len(label))
      maxlength = str(maxlength)

      # Header line
      fmt = "%-"+maxlength+"s"
      lineDesc = data.getAxisDescriptionHeader()
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
         self._printLine(y[:,i], maxlength, "int")

      # Print count stats
      for func in [Common.nanmin, Common.nanmax]:
         name = func.__name__[3:]
         print lineDescFmt % ("num " + name),
         values = np.zeros(F, 'float')
         for f in range(0,F):
            values[f] = np.sum(y[f,:] == func(y,axis=0))
         self._printLine(values, maxlength, "int")

   def _printLine(self, values , colWidth, type="float"):
      if(type == "int"):
         fmt  = "%-"+colWidth+"i"
      else:
         fmt         = "%-"+colWidth+".2f"
      missfmt = "%-"+colWidth+"s"
      minI    = np.argmin(values)
      maxI    = np.argmax(values)
      for f in range(0, len(values)):
         value = values[f]
         if(np.isnan(value)):
            txt = missfmt % "--"
         else:
            txt = fmt % value
         if(minI == f):
            print Common.green(txt),
         elif(maxI == f):
            print Common.red(txt),
         else:
            print txt,
      print ""

class Sort(Output):
   _reqThreshold = False
   _supThreshold = False
   def __init__(self, name):
      Output.__init__(self)
      self._name = name

   def _plotCore(self, data):
      data.setAxis("none")
      labels = self._getLegendNames(data)
      F = data.getNumFiles()
      for f in range(0, F):
         data.setFileIndex(f)
         [x] = data.getScores(self._name)
         x = np.sort(x)
         color = self._getColor(f, F)
         style = self._getStyle(f, F)
         y = np.linspace(0, 1, x.shape[0])
         mpl.plot(x, y, style, color=color, label=labels[f], lw=self._lw, ms=self._ms)
      mpl.xlabel("Sorted " + data.getAxisLabel("threshold"))
      mpl.grid()


class ObsFcst(Output):
   _supThreshold = False
   _description = "Plot observations and forecasts"
   def __init__(self):
      Output.__init__(self)
   def _plotCore(self, data):
      F = data.getNumFiles()
      data.setAxis(self._xaxis)
      x = data.getAxisValues()

      isCont = data.isAxisContinuous()

      # Obs line
      mObs  = Metric.Mean(Metric.Default("obs"))
      y = mObs.compute(data, None)
      self._plotObs(x, y, isCont)

      mFcst = Metric.Mean(Metric.Default("fcst"))
      labels = data.getFilenames()
      for f in range(0, F):
         data.setFileIndex(f)
         color = self._getColor(f, F)
         style = self._getStyle(f, F, isCont)

         y = mFcst.compute(data, None)
         mpl.plot(x, y, style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
      mpl.ylabel(data.getVariableAndUnits())
      mpl.xlabel(data.getAxisLabel())
      mpl.grid()
      mpl.gca().xaxis.set_major_formatter(data.getAxisFormatter())

class QQ(Output):
   _supThreshold = False
   _supX = False
   _description = "Quantile-quantile plot of obs vs forecasts"
   def __init__(self):
      Output.__init__(self)
   def getXY(self, data):
      x = list()
      y = list()
      F = len(data.getFilenames())
      for f in range(0, F):
         data.setFileIndex(f)
         [xx,yy] = data.getScores(["obs", "fcst"])
         x.append(np.sort(xx))
         y.append(np.sort(yy))
      return [x,y]

   def _plotCore(self, data):
      data.setAxis("none")
      data.setIndex(0)
      labels = data.getFilenames()
      F = data.getNumFiles()
      [x,y] = self.getXY(data)
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F)

         mpl.plot(x[f], y[f], style, color=color, label=labels[f], lw=self._lw,
               ms=self._ms)
      mpl.ylabel("Sorted forecasts (" + data.getUnits() + ")")
      mpl.xlabel("Sorted observations (" + data.getUnits() + ")")
      ylim = list(mpl.ylim())
      xlim = list(mpl.xlim())
      axismin = min(min(ylim),min(xlim))
      axismax = max(max(ylim),max(xlim))
      #mpl.plot([axismin,axismax], [axismin,axismax], "--", color=[0.3,0.3,0.3], lw=3, zorder=-100)
      self._plotPerfectScore([axismin,axismax], [axismin,axismax])
      mpl.grid()
   def _textCore(self, data):
      data.setAxis("none")
      data.setIndex(0)
      labels = data.getFilenames()
      F = data.getNumFiles()

      # Header
      maxlength = 0
      for name in data.getFilenames():
         maxlength = max(maxlength, len(name))
      maxlength = int(np.ceil(maxlength/2)*2)
      fmt = "%"+str(maxlength)+"s"
      for filename in data.getFilenames():
         print fmt % filename,
      print ""
      fmt = "%" + str(int(np.ceil(maxlength/2))) + ".1f"
      fmt = fmt + fmt
      fmtS = "%" + str(int(np.ceil(maxlength/2))) + "s"
      fmtS = fmtS + fmtS
      for f in range(0, F):
         print fmtS % ("obs", "fcst"),
      print ""

      [x,y] = self.getXY(data)
      maxPairs = len(x[0])
      for f in range(1, F):
         maxPairs = max(maxPairs, len(x[f]))
      for i in range(0, maxPairs):
         for f in range(0, F):
            if(len(x[f]) < i):
               print " --  -- "
            else:
               print fmt % (x[f][i], y[f][i]),
         print "\n",

class Scatter(Output):
   _description = "Scatter plot of forecasts vs obs"
   _supThreshold = False
   _supX = False
   def __init__(self):
      Output.__init__(self)
   def _plotCore(self, data):
      data.setAxis("none")
      data.setIndex(0)
      labels = data.getFilenames()
      F = data.getNumFiles()
      for f in range(0, F):
         data.setFileIndex(f)
         color = self._getColor(f, F)
         style = self._getStyle(f, F, connectingLine=False)

         [x,y] = data.getScores(["obs","fcst"])
         mpl.plot(x,y, ".", color=color, label=labels[f], lw=self._lw,
               ms=self._ms, alpha=0.2)
      mpl.ylabel("Forecasts (" + data.getUnits() + ")")
      mpl.xlabel("Observations (" + data.getUnits() + ")")
      ylim = mpl.ylim()
      xlim = mpl.xlim()
      axismax = max(max(ylim),max(xlim))
      mpl.plot([0,axismax], [0,axismax], "--", color=[0.3,0.3,0.3], lw=3, zorder=-100)
      mpl.grid()

class Change(Output):
   _supThreshold = False
   _supX = False
   _description = "Forecast skill (MAE) as a function of change in obs from previous day"
   def __init__(self):
      Output.__init__(self)

   def _plotCore(self, data):
      data.setAxis("all")
      data.setIndex(0)
      labels = data.getFilenames()
      # Find range
      data.setFileIndex(0)
      [obs,fcst] = data.getScores(["obs", "fcst"])
      change = obs[1:,Ellipsis]-obs[0:-1,Ellipsis]
      maxChange = np.nanmax(abs(change.flatten()))
      edges = np.linspace(-maxChange,maxChange,20)
      bins  = (edges[1:] + edges[0:-1])/2
      F = data.getNumFiles()

      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F)
         data.setFileIndex(f)
         [obs,fcst] = data.getScores(["obs", "fcst"])
         change = obs[1:,Ellipsis]-obs[0:-1,Ellipsis]
         err = abs(obs-fcst)
         err = err[1:,Ellipsis]
         x = np.nan * np.zeros(len(bins), 'float')
         y = np.nan * np.zeros(len(bins), 'float')

         for i in range(0, len(bins)):
            I = (change > edges[i] ) & (change <= edges[i+1])
            y[i] = Common.nanmean(err[I])
            x[i] = Common.nanmean(change[I])
         mpl.plot(x, y, style, color=color, lw=self._lw, ms=self._ms, label=labels[f])
      self._plotPerfectScore(x, 0)
      mpl.xlabel("Daily obs change (" + data.getUnits() + ")")
      mpl.ylabel("MAE (" + data.getUnits() + ")")
      mpl.grid()

class Cond(Output):
   _description = "Plots forecasts as a function of obs (use -r to specify bin-edges)"
   _defaultAxis = "threshold"
   _defaultBinType = "within"
   _reqThreshold = True
   _supThreshold = True
   _supX = False
   def supportsThreshold(self):
      return True
   def _plotCore(self, data):
      data.setAxis("none")
      data.setIndex(0)
      [lowerT,upperT,x] = self._getThresholdLimits(self._thresholds)

      labels = data.getFilenames()
      F = data.getNumFiles()
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F)
         data.setFileIndex(f)

         of = np.zeros(len(x), 'float')
         fo = np.zeros(len(x), 'float')
         xof = np.zeros(len(x), 'float')
         xfo = np.zeros(len(x), 'float')
         mof = Metric.Conditional("obs", "fcst", np.mean) # F | O
         mfo = Metric.Conditional("fcst", "obs", np.mean) # O | F
         xmof = Metric.XConditional("obs", "fcst") # F | O
         xmfo = Metric.XConditional("fcst", "obs") # O | F
         mof0 = Metric.Conditional("obs", "fcst", np.mean) # F | O
         for i in range(0, len(lowerT)):
            fo[i] = mfo.compute(data, [lowerT[i], upperT[i]])
            of[i] = mof.compute(data, [lowerT[i], upperT[i]])
            xfo[i] = xmfo.compute(data, [lowerT[i], upperT[i]])
            xof[i] = xmof.compute(data, [lowerT[i], upperT[i]])
         mpl.plot(xof,of, style, color=color, label=labels[f] + " (F|O)", lw=self._lw, ms=self._ms)
         mpl.plot(fo, xfo, style, color=color, label=labels[f] + " (O|F)", lw=self._lw, ms=self._ms, alpha=0.5)
      mpl.ylabel("Forecasts (" + data.getUnits() + ")")
      mpl.xlabel("Observations (" + data.getUnits() + ")")
      ylim = mpl.ylim()
      xlim = mpl.xlim()
      axismin = min(min(ylim),min(xlim))
      axismax = max(max(ylim),max(xlim))
      #mpl.plot([axismin,axismax], [axismin,axismax], "-", color="k", lw=3, zorder=-100)
      self._plotPerfectScore([axismin,axismax], [axismin,axismax])
      mpl.grid()

class SpreadSkill(Output):
   _supThreshold = False
   _supX = False
   _description = "Spread skill"
   def __init__(self):
      Output.__init__(self)

   def _plotCore(self, data):
      data.setAxis("all")
      data.setIndex(0)
      labels = data.getFilenames()
      F = data.getNumFiles()
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F, connectingLine=False)
         data.setFileIndex(f)

         data.setFileIndex(f)
         [obs,fcst,spread] = data.getScores(["obs", "fcst", "ensSpread"])
         spread = np.sqrt(spread.flatten())
         skill = abs(obs.flatten()- fcst.flatten())
         #mpl.plot(spread, skill, style, color=color, lw=self._lw, ms=self._ms, label=labels[f])
         x = np.zeros(len(self._thresholds), 'float')
         y = np.zeros(len(x), 'float')
         for i in range(1, len(self._thresholds)):
            I = np.where((np.isnan(spread) == 0) &
                         (np.isnan(skill) == 0) &
                         (spread > self._thresholds[i-1]) &
                         (spread <= self._thresholds[i]))[0]
            if(len(I) > 0):
               x[i] = np.mean(spread[I])
               y[i] = np.mean(skill[I])

         style = self._getStyle(f, F)
         mpl.plot(x, y, style, color=color, label=labels[f])
      mpl.xlabel("Spread (" + data.getUnits() + ")")
      mpl.ylabel("MAE (" + data.getUnits() + ")")
      mpl.grid()

class Count(Output):
   _description = "Counts number of forecasts above or within thresholds (use -r to specify bin-edges). Use -binned to count number in bins, instead of number above each threshold."
   _defaultAxis = "threshold"
   _defaultBinType = "within"
   _reqThreshold = True
   _supThreshold = True
   _supX = False
   def _plotCore(self, data):
      data.setAxis("none")
      data.setIndex(0)
      [lowerT,upperT,x] = self._getThresholdLimits(self._thresholds)

      labels = data.getFilenames()
      F = data.getNumFiles()
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F)
         data.setFileIndex(f)

         Nobs = np.zeros(len(x), 'float')
         Nfcst = np.zeros(len(x), 'float')
         obs  = Metric.Count("obs")
         fcst = Metric.Count("fcst")
         for i in range(0, len(lowerT)):
            Nobs[i] = obs.compute(data, [lowerT[i], upperT[i]])
            Nfcst[i] = fcst.compute(data, [lowerT[i], upperT[i]])
         mpl.plot(x,Nfcst, style, color=color, label=labels[f], lw=self._lw, ms=self._ms)
      self._plotObs(x, Nobs)
      mpl.ylabel("Number")
      mpl.xlabel(data.getAxisLabel())
      mpl.grid()

class TimeSeries(Output):
   _description = "Plot observations and forecasts as a time series (i.e. by concatinating all offsets). '-x <dimension>' has no effect, as it is always shown by date."
   _supThreshold = False
   _supX = False
   def _plotCore(self, data):
      F = data.getNumFiles()
      data.setAxis("all")
      dates = data.getAxisValues("date")
      offsets = data.getAxisValues("offset")

      # Connect the last offset of a day with the first offset on the next day
      # This only makes sense if the obs/fcst don't span more than a day
      connect = min(offsets) + 24 > max(offsets)
      minOffset = min(offsets)

      # Obs line
      obs = data.getScores("obs")[0]
      for d in range(0,obs.shape[0]):
         x = dates[d] + offsets/24.0
         y = Common.nanmean(obs[d,:,:], axis=1)
         if(connect and d < obs.shape[0]-1):
            x = np.insert(x,x.shape[0],dates[d+1]+minOffset/24.0)
            y = np.insert(y,y.shape[0],Common.nanmean(obs[d+1,0,:], axis=0))

         if(d==0): # tchui (10/06/15)
            xmin=np.min(x)
         elif(d==obs.shape[0]-1):
            xmax=np.max(x)

         lab = "obs" if d == 0 else ""
         mpl.rcParams['ytick.major.pad']='20'         ######This changes the buffer zone between tick labels and the axis. (dsiuta)
         #mpl.rcParams['ytick.major.pad']='${self._pad}'
         #mpl.rcParams['xtick.major.pad']='${self._pad}'
         mpl.rcParams['xtick.major.pad']='20'         ######This changes the buffer zone between tick labels and the axis. (dsiuta)
         mpl.plot(x, y,  ".-", color=[0.3,0.3,0.3], lw=5, label=lab)

         # Forecast lines
         labels = data.getFilenames()
         for f in range(0, F):
            data.setFileIndex(f)
            color = self._getColor(f, F)
            style = self._getStyle(f, F)

            fcst = data.getScores("fcst")[0]
            x = dates[d] + offsets/24.0
            y = Common.nanmean(fcst[d,:,:], axis=1)
            if(connect and d < obs.shape[0]-1):
               x = np.insert(x,x.shape[0],dates[d+1]+minOffset/24.0)
               y = np.insert(y,y.shape[0],Common.nanmean(fcst[d+1,0,:]))
            lab = labels[f] if d == 0 else ""
            mpl.rcParams['ytick.major.pad']='20'  ######This changes the buffer zone between tick labels and the axis. (dsiuta)
            mpl.rcParams['xtick.major.pad']='20'    ######This changes the buffer zone between tick labels and the axis. (dsiuta)
            #mpl.rcParams['ytick.major.pad']='${self._pad}'
            #mpl.rcParams['xtick.major.pad']='${self._pad}'
            mpl.plot(x, y,  style, color=color, lw=self._lw, ms=self._ms, label=lab)


      #mpl.ylabel(data.getVariableAndUnits())  # "Wind Speed (km/hr)") ###hard coded axis label (dsiuta)
      mpl.xlabel(data.getAxisLabel("date"))
      if(self._ylabel == None):
         mpl.ylabel(data.getVariableAndUnits())
      else:
         mpl.ylabel(self._ylabel)
     # mpl.ylabel(self._ylabel)  # "Wind Speed (km/hr)") ###hard coded axis label (dsiuta)
      mpl.grid()
      mpl.gca().xaxis.set_major_formatter(data.getAxisFormatter("date"))

      if(self._tight):
         oldTicks=mpl.gca().get_xticks()
         diff = oldTicks[1] - oldTicks[0] # keep auto tick interval
         tickRange = np.arange(round(xmin),round(xmax)+diff,diff)
         mpl.gca().set_xticks(tickRange) # make new ticks, to start from the first day of the desired interval
         mpl.autoscale(enable=True,axis=u'x',tight=True) # make xaxis tight



class PitHist(Output):
   _description = "Histogram of PIT values"
   _supThreshold = False
   _supX = False
   def __init__(self, metric):
      Output.__init__(self)
      self._numBins = 10
      self._metric = metric
   def _legend(self, data,names=None):
      pass
   def _plotCore(self, data):
      F = data.getNumFiles()
      labels = self._getLegendNames(data)
      for f in range(0, F):
         Common.subplot(f,F)
         color = self._getColor(f, F)
         data.setAxis("none")
         data.setIndex(0)
         data.setFileIndex(f)
         pit = self._metric.compute(data,None)

         width = 1.0 / self._numBins
         x = np.linspace(0,1,self._numBins+1)
         N = np.histogram(pit, x)[0]
         n = N * 1.0 / sum(N)
         color = "gray"
         xx = x[range(0,len(x)-1)]
         mpl.bar(xx, n*100.0, width=width, color=color)
         mpl.plot([0,1],[100.0/self._numBins, 100.0/self._numBins], 'k--')
         #self._plotPerfectScore([0,1],[100.0/self._numBins, 100.0/self._numBins], "r", 100)
         mpl.title(labels[f]);
         ytop = 200.0/self._numBins
         mpl.gca().set_ylim([0,ytop])
         if(f == 0):
            mpl.ylabel("Frequency (%)")
         else:
            mpl.gca().set_yticks([])

         # Multiply by 100 to get to percent
         std = Metric.PitDev.deviationStd(pit, self._numBins)*100

         mpl.plot([0,1], [100.0/self._numBins - 2*std, 100.0/self._numBins - 2*std], "r-")
         mpl.plot([0,1], [100.0/self._numBins + 2*std, 100.0/self._numBins + 2*std], "r-")
         Common.fill([0,1], [100.0/self._numBins - 2*std, 100.0/self._numBins - 2*std], [100.0/self._numBins
            + 2*std, 100.0/self._numBins + 2*std], "r", zorder=100, alpha=0.5)

         # Compute calibration deviation
         D  = Metric.PitDev.deviation(pit, self._numBins)
         D0 = Metric.PitDev.expectedDeviation(pit, self._numBins)
         ign = Metric.PitDev.ignorancePotential(pit, self._numBins)
         mpl.text(0, mpl.ylim()[1], "Dev: %2.4f\nExp: %2.4f\nIgn: %2.4f" % (D,D0,ign), verticalalignment="top")

         mpl.xlabel("Cumulative probability")

class Reliability(Output):
   _description = "Reliability diagram for a certain threshold (-r)"
   _reqThreshold = True
   _supX = False
   _legLoc = "lower right"
   def __init__(self):
      Output.__init__(self)
      self._shadeNoSkill = True
   def _plotCore(self, data):
      labels = data.getFilenames()

      F = data.getNumFiles()
      ax  = mpl.gca()
      axi = mpl.axes([0.2,0.65,0.2,0.2])
      mpl.sca(ax)

      data.setAxis("none")
      data.setIndex(0)
      data.setFileIndex(0)
      for t in range(0,len(self._thresholds)):
         threshold = self._thresholds[t]
         var = data.getPvar(threshold)
         [obs, p] = data.getScores(["obs", var])

         # Determine the number of bins to use # (at least 11, at most 25)
         N = min(25, max(11, int(len(obs)/1000)))
         N = 11
         edges = np.linspace(0,1,N+1)
         edges = np.array([0,0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1])
         #edges = np.linspace(0,1,101)
         x  = np.zeros([len(edges)-1,F], 'float')

         y = np.nan*np.zeros([F,len(edges)-1],'float')
         n = np.zeros([F,len(edges)-1],'float')
         v = np.zeros([F,len(edges)-1],'float') # Variance
         # Draw reliability lines
         for f in range(0, F):
            color = self._getColor(f, F)
            style = self._getStyle(f, F)
            data.setFileIndex(f)
            data.setAxis("none")
            data.setIndex(0)
            var = data.getPvar(threshold)
            [obs, p] = data.getScores(["obs", var])

            if(self._binType == "below"):
               p = p
               obs = obs < threshold
            elif(self._binType == "above"):
               p = 1 - p
               obs = obs > threshold
            else:
               Common.error("Bin type must be one of 'below' or 'above' for reliability plot")

            clim = np.mean(obs)
            # Compute frequencies
            for i in range(0,len(edges)-1):
               q = (p >= edges[i])& (p < edges[i+1])
               I = np.where(q)[0]
               if(len(I) > 0):
                  n[f,i] = len(obs[I])
                  # Need at least 10 data points to be valid
                  if(n[f,i] >= 1):
                     y[f,i] = np.mean(obs[I])
                     v[f,i] = np.var(obs[I])
                  x[i,f] = np.mean(p[I])

            label = labels[f]
            if(not t == 0):
               label = ""
            mpl.plot(x[:,f], y[f], style, color=color, lw=self._lw, ms=self._ms, label=label)

         # Draw confidence bands (do this separately so that these lines don't sneak into the legend)
         for f in range(0, F):
            color = self._getColor(f, F)
            self._plotConfidence(x[:,f], y[f], v[f], n[f], color=color)

         # Draw lines in inset diagram
         if(np.max(n) > 1):
            for f in range(0, F):
               color = self._getColor(f, F)
               axi.plot(x[:,f], n[f], style, color=color, lw=self._lw, ms=self._ms*0.75)
            axi.xaxis.set_major_locator(mpl.NullLocator())
            axi.set_yscale('log')
            axi.set_title("Number")
            axi.grid('on')
      mpl.sca(ax)
      self._plotObs([0,1], [0,1])
      mpl.axis([0,1,0,1])
      color = "gray"
      mpl.plot([0,1], [clim,clim], "--", color=color,label="")  # Climatology line
      mpl.plot([clim,clim], [0,1], "--", color=color)           # Climatology line
      mpl.plot([0,1], [clim/2,1-(1-clim)/2], "--", color=color) # No-skill line
      if(self._shadeNoSkill):
         Common.fill([clim,1], [0,0], [clim,1-(1-clim)/2], col=[1,1,1], zorder=-100,
               hatch="\\")
         Common.fill([0,clim], [clim/2,clim,0], [1,1], col=[1,1,1], zorder=-100,
               hatch="\\")
      mpl.xlabel("Forecasted probability")
      mpl.ylabel("Observed frequency")
      units = " " + data.getUnits()
      mpl.title("Reliability diagram for obs > " + str(threshold) + units)

class IgnContrib(Output):
   _description = "Binary Ignorance contribution diagram for a single threshold (-r). "\
         + "Shows how much each probability issued contributes to the total ignorance."
   _reqThreshold = True
   _supX = False
   _legLoc = "upper center"
   _experimental = True
   def __init__(self):
      Output.__init__(self)
   def _plotCore(self, data):
      labels = data.getFilenames()

      if(len(self._thresholds) != 1):
         Common.error("IgnContrib diagram requires exactly one threshold")
      threshold = self._thresholds[0]

      F = data.getNumFiles()

      mpl.subplot(2,1,1)
      units = " " + data.getUnits()
      titlestr = "Ignorance contribution diagram for obs > " + str(self._thresholds[0]) + units
      mpl.title(titlestr)

      data.setAxis("none")
      data.setIndex(0)
      data.setFileIndex(0)
      mpl.subplot(2,1,1)
      var = data.getPvar(threshold)
      [obs, p] = data.getScores(["obs", var])

      # Determine the number of bins to use # (at least 11, at most 25)
      N = min(25, max(11, int(len(obs)/1000)))
      edges = np.linspace(0,1,N+1)

      x  = np.zeros([F, len(edges)-1], 'float')
      y = np.nan*np.zeros([F,len(edges)-1],'float')
      n = np.zeros([F,len(edges)-1],'float')

      # Draw reliability lines
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F)
         data.setFileIndex(f)
         data.setAxis("none")
         data.setIndex(0)
         var = data.getPvar(threshold)
         [obs, p] = data.getScores(["obs", var])

         if(self._binType == "below"):
            p = p
            obs = obs < threshold
         elif(self._binType == "above"):
            p = 1 - p
            obs = obs > threshold
         else:
            Common.error("Bin type must be one of 'below' or 'above' for reliability plot")

         clim = np.mean(obs)
         # Compute frequencies
         for i in range(0,len(edges)-1):
            q = (p >= edges[i])& (p < edges[i+1])
            I = np.where(q)[0]
            if(len(I) > 0):
               n[f,i] = len(obs[I])
               x[f,i] = np.mean(p[I])
               # Need at least 10 data points to be valid
               if(n[f,i] >= 1):
                  #y[f,i] = -n[f,i]*(x[f,i]*np.log2(x[f,i]) + (1-x[f,i])*np.log2(1-x[f,i]))
                  I0 = np.where(obs[I] == 0)
                  I1 = np.where(obs[I] == 1)
                  y[f,i] = -np.sum(np.log2(p[I[I1]])) - np.sum(np.log2(1-p[I[I0]]))

         label = labels[f]
         mpl.plot(x[f], y[f]/np.sum(n[f])*len(n[f]), style, color=color, lw=self._lw, ms=self._ms, label=label)
      mpl.ylabel("Ignorance contribution")

      # Draw expected sharpness
      xx = np.linspace(0,1,100)
      yy = -(xx*np.log2(xx) + (1-xx)*np.log2(1-xx))
      mpl.plot(xx, yy, "--", color="gray")
      yy = -np.log2(clim)*np.ones(len(xx))

      # Show number in each bin
      mpl.subplot(2,1,2)
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F)
         mpl.plot(x[f], n[f], style, color=color, lw=self._lw, ms=self._ms)
      mpl.xlabel("Forecasted probability")
      mpl.ylabel("N")

      # Switch back to top subpplot, so the legend works
      mpl.subplot(2,1,1)

# doClassic: Use the classic definition, by not varying the forecast threshold
#            i.e. using the same threshold for observation and forecast.
class DRoc(Output):
   _description = "Plots the receiver operating characteristics curve for the deterministic " \
         + "forecast for a single threshold. Uses different forecast thresholds to create points."
   _supX = False
   _reqThreshold = True
   def __init__(self, fthresholds=None, doNorm=False, doClassic=False):
      Output.__init__(self)
      self._doNorm = doNorm
      self._fthresholds = fthresholds
      self._doClassic = doClassic
      self._showThresholds = False
   def _plotCore(self, data):
      threshold = self._thresholds[0]   # Observation threshold
      if(threshold == None):
         Common.error("DRoc plot needs a threshold (use -r)")

      if(self._doClassic):
         fthresholds = [threshold]
      else:
         if(self._fthresholds != None):
            fthresholds = self._fthresholds
         else:
            if(data.getVariable() == "Precip"):
               fthresholds = [0,1e-7,1e-6,1e-5,1e-4,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.5,1,2,3,5,10,20,100]
            else:
               N = 31
               fthresholds = np.linspace(threshold-10, threshold+10, N)

      F = data.getNumFiles()
      labels = data.getFilenames()
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F)
         data.setAxis("none")
         data.setIndex(0)
         data.setFileIndex(f)
         [obs, fcst] = data.getScores(["obs", "fcst"])

         y = np.nan*np.zeros([len(fthresholds),1],'float')
         x = np.nan*np.zeros([len(fthresholds),1],'float')
         for i in range(0,len(fthresholds)):
            fthreshold = fthresholds[i]
            a    = np.ma.sum((fcst >= fthreshold) & (obs >= threshold)) # Hit
            b    = np.ma.sum((fcst >= fthreshold) & (obs <  threshold)) # FA
            c    = np.ma.sum((fcst <  fthreshold) & (obs >= threshold)) # Miss
            d    = np.ma.sum((fcst <  fthreshold) & (obs <  threshold)) # Correct rejection
            if(a + c > 0 and b + d > 0):
               y[i] = a / 1.0 / (a + c)
               x[i] = b / 1.0 / (b + d)
               if(self._doNorm):
                  from scipy.stats import norm
                  y[i] = norm.ppf(a / 1.0 / (a + c))
                  x[i] = norm.ppf(b / 1.0 / (b + d))
                  if(np.isinf(y[i])):
                     y[i] = np.nan
                  if(np.isinf(x[i])):
                     x[i] = np.nan
               if(self._showThresholds and (not np.isnan(x[i]) and not np.isnan(y[i]) and f == 0)):
                  mpl.text(x[i], y[i], "%2.1f" % fthreshold, color=color)
         if(not self._doNorm):
            # Add end points at 0,0 and 1,1:
            xx = x
            yy = y
            x = np.zeros([len(fthresholds)+2,1], 'float')
            y = np.zeros([len(fthresholds)+2,1], 'float')
            x[1:-1] = xx
            y[1:-1] = yy
            x[0] = 1
            y[0] = 1
            x[len(x)-1] = 0
            y[len(y)-1] = 0
         #I = np.where(np.isnan(x)+np.isnan(y)==0)
         mpl.plot(x, y, style, color=color, label=labels[f], lw=self._lw, ms=self._ms)
      if(self._doNorm):
         xlim = mpl.xlim()
         ylim = mpl.ylim()
         q0 =  max(abs(xlim[0]), abs(ylim[0]))
         q1 =  max(abs(xlim[1]), abs(ylim[1]))
         mpl.plot([-q0,q1], [-q0,q1], 'k--')
         mpl.xlabel("Normalized false alarm rate")
         mpl.ylabel("Normalized hit rate")
      else:
         mpl.plot([0,1], [0,1], color="k")
         mpl.axis([0,1,0,1])
         mpl.xlabel("False alarm rate")
         mpl.ylabel("Hit rate")
         self._plotPerfectScore([0,0,1], [0,1,1])
      units = " " + data.getUnits()
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
   _defaultAxis = "none"
   _supThreshold = False
   _supX = False
   _minStdDiff = 0.1 # How big difference should colour kick in (in number of STDs)?
   def _plotCore(self, data):
      F = data.getNumFiles()
      if(F < 2):
         Common.error("Cannot use Against plot with less than 2 configurations")

      data.setAxis("none")
      data.setIndex(0)
      labels = data.getFilenames()
      for f0 in range(0,F):
         for f1 in range(0,F):
            if(f0 != f1 and (F != 2 or f0 == 0)):
               if(F > 2):
                  mpl.subplot(F,F,f0+f1*F+1)
               data.setFileIndex(f0)
               x = data.getScores("fcst")[0].flatten()
               data.setFileIndex(f1)
               y = data.getScores("fcst")[0].flatten()
               lower = min(min(x),min(y))
               upper = max(max(x),max(y))

               mpl.plot(x, y, "x", mec="k", ms=self._ms/2, mfc="k", zorder=-1000)

               # Show which forecast is better
               data.setFileIndex(f0)
               [obsx,x] = data.getScores(["obs","fcst"])
               data.setFileIndex(f1)
               [obsy,y] = data.getScores(["obs","fcst"])
               x = x.flatten()
               y = y.flatten()
               obs = obsx.flatten()

               mpl.plot(x, y, "s", mec="k", ms=self._ms/2, mfc="w", zorder=-500)

               std = np.std(obs)/2
               minDiff = self._minStdDiff*std
               if(len(x) == len(y)):
                  N = 5
                  for k in range(0,N):
                     Ix = abs(obs - y) > abs(obs - x) + std*k/N
                     Iy = abs(obs - y) + std*k/N < abs(obs - x)
                     mpl.plot(x[Ix], y[Ix], "r.", ms=self._ms, alpha=k/1.0/N)
                     mpl.plot(x[Iy], y[Iy], "b.", ms=self._ms, alpha=k/1.0/N)

               # Contour of the frequency
               #q = np.histogram2d(x[1,:], x[0,:], [np.linspace(lower,upper,100), np.linspace(lower,upper,100)])
               #[X,Y] = np.meshgrid(q[1],q[2])
               #mpl.contour(X[1:,1:],Y[1:,1:],q[0],[1,100],zorder=90)

               mpl.xlabel(labels[f0], color="r")
               mpl.ylabel(labels[f1], color="b")
               mpl.grid()
               xlim = mpl.xlim()
               ylim = mpl.ylim()
               lower = min(xlim[0],ylim[0])
               upper = max(xlim[1],ylim[1])
               mpl.xlim([lower, upper])
               mpl.ylim([lower, upper])
               mpl.plot([lower,upper], [lower, upper], '--', color=[0.3,0.3,0.3], lw=3, zorder=100)
               if(F == 2):
                  break
   def _legend(self, data, names=None):
      pass

class Taylor(Output):
   _description = "Taylor diagram showing correlation and forecast standard deviation"
   _supThreshold = False
   _supX = False
   _defaultAxis = "none"
   _legLoc = "upper left"

   def _plotCore(self, data):
      data.setAxis(self._xaxis)
      data.setIndex(0)
      labels = data.getFilenames()
      F = data.getNumFiles()

      # Plot points
      maxstd = 0
      for f in range(0, F):
         data.setFileIndex(f)
         color = self._getColor(f, F)
         style = self._getStyle(f, F)

         size   = data.getAxisSize()
         corr   = np.zeros(size, 'float')
         std    = np.zeros(size, 'float')
         stdobs = np.zeros(size, 'float')
         for i in range(0,size):
            data.setIndex(i)
            [obs,fcst] = data.getScores(["obs", "fcst"])
            if(len(obs)>0 and len(fcst)>0):
               corr[i] = np.corrcoef(obs,fcst)[1,0]
               std[i]  = np.sqrt(np.var(fcst))
               stdobs[i] = np.sqrt(np.var(obs))
         maxstd = max(maxstd, max(std))
         ang = np.arccos(corr)
         x = std * np.cos(ang)
         y = std * np.sin(ang)
         mpl.plot(x,y, style, color=color, label=labels[f], lw=self._lw, ms=self._ms)
         stdobs = np.mean(stdobs)

      # Set axis limits
      if(maxstd < 1.25*stdobs): # Enforce a minimum radius beyond the obs-radius
         maxstd = 1.25*stdobs
      maxstd = int(np.ceil(maxstd))
      mpl.xlim([-maxstd*1.05,maxstd*1.05]) # Allow for some padding outside the outer ring
      mpl.ylim([0,maxstd*1.05])
      mpl.xlabel("Standard deviation (" + data.getUnits() + ")")
      xticks = mpl.xticks()[0]
      mpl.xticks(xticks[xticks>=0])
      mpl.xlim([-maxstd*1.05,maxstd*1.05])
      mpl.ylim([0,maxstd*1.05])
      mpl.text(np.sin(np.pi/4)*maxstd, np.cos(np.pi/4)*maxstd, "Correlation", rotation=-45,
            fontsize=self._labfs, horizontalalignment="center", verticalalignment="bottom")
      mpl.gca().yaxis.set_visible(False)
      mpl.gca().xaxis.set_ticks_position('bottom')

      # Draw obs point/lines
      orange = [1,0.73,0.2]
      self._drawCircle(stdobs, style='-', lw=2, color=orange)
      mpl.plot(stdobs, 0, 's-', color=orange, label="Observation", lw=self._lw, ms=self._ms)
      mpl.plot([-maxstd, maxstd], [0,0], '-', color=orange, lw=self._lw*2)

      # Draw diagonals
      corrs = [-1,-0.99,-0.95,-0.9,-0.8,-0.5,0,0.5,0.8,0.9,0.95,0.99] #np.linspace(-1,1,21)
      for i in range(0, len(corrs)):
         ang = np.arccos(corrs[i]) # Mathematical angle
         x = np.cos(ang)*maxstd
         y = np.sin(ang)*maxstd
         mpl.plot([0, x], [0, y], 'k--')
         mpl.text(x, y, str(corrs[i]), verticalalignment="bottom", fontsize=self._labfs)

      # Draw CRMSE rings
      xticks = mpl.xticks()[0]
      self._drawCircle(0,style="-", color="gray", lw=3, label="CRMSE")
      for R in np.linspace(0, 2*max(xticks), 2*2*max(xticks)/(xticks[1]-xticks[0])+1):
         if(R > 0):
            self._drawCircle(R, xcenter=stdobs, ycenter=0, maxradius=maxstd, style="-", color="gray", lw=3)
            x = np.sin(-np.pi/4)*R+stdobs
            y = np.cos(np.pi/4)*R
            if(x**2+y**2 < maxstd**2):
               mpl.text(x, y, str(R), horizontalalignment="right", verticalalignment="bottom",
                     fontsize=self._labfs, color="gray")

      # Draw std rings
      for X in mpl.xticks()[0]:
         if(X <= maxstd):
            self._drawCircle(X, style=":")
      self._drawCircle(maxstd, style="-", lw=3)

      mpl.gca().set_aspect(1)

class Error(Output):
   _description = "Decomposition of RMSE into systematic and unsystematic components"
   _supThreshold = False
   _supX = True
   _defaultAxis = "none"

   def _plotCore(self, data):
      data.setAxis(self._xaxis)
      data.setIndex(0)
      labels = data.getFilenames()
      F = data.getNumFiles()

      mpl.gca().set_aspect(1)
      mpl.xlabel("Unsystematic error (CRMSE, " + data.getUnits() + ")")
      mpl.ylabel("Systematic error (Bias, " + data.getUnits() + ")")


      # Plot points
      size   = data.getAxisSize()
      serr   = np.nan*np.zeros([size,F], 'float')
      uerr   = np.nan*np.zeros([size,F], 'float')
      rmse   = np.nan*np.zeros([size,F], 'float')
      for f in range(0, F):
         data.setFileIndex(f)
         color = self._getColor(f, F)
         style = self._getStyle(f, F, connectingLine=False)

         for i in range(0,size):
            data.setIndex(i)
            [obs,fcst] = data.getScores(["obs", "fcst"])
            mfcst = np.mean(fcst)
            mobs  = np.mean(obs)
            if(len(obs)>0 and len(fcst)>0):
               serr[i,f] = np.mean(obs-fcst)
               rmse[i,f] = np.sqrt(np.mean((obs-fcst)**2))
               uerr[i,f] = np.sqrt(rmse[i,f]**2 - serr[i,f]**2)
            # np.sqrt(np.mean((fcst - mfcst) - (obs - mobs))**2)
         mpl.plot(uerr[:,f],serr[:,f], style, color=color, label=labels[f], lw=self._lw, ms=self._ms)
      xlim = mpl.xlim()
      ylim = mpl.ylim()

      # Draw rings
      for f in range(0, F):
         color = self._getColor(f, F)
         style = self._getStyle(f, F, lineOnly=True)
         self._drawCircle(Common.nanmean(rmse[:,f]), style=style, color=color)

      # Set axis limits
      maxx = xlim[1]
      maxy = ylim[1]
      miny = min(0,ylim[0])
      # Try to enforce the x-axis and y-axis to be roughly the same size
      if(maxy - miny < maxx/2):
         maxy = maxx
      elif(maxy - miny > maxx*2):
         maxx = maxy-miny
      mpl.xlim([0, maxx]) # Not possible to have negative CRMSE
      mpl.ylim([miny,maxy])

      # Draw standard RMSE rings
      for X in mpl.xticks()[0]:
         self._drawCircle(X, style=":")

      mpl.plot([0,maxx], [0,0], 'k-', lw=2) # Draw x-axis line
      mpl.grid()

class Marginal(Output):
   _description = "Show marginal distribution for different thresholds"
   _reqThreshold = True
   _supX = False
   _experimental = True
   def __init__(self):
      Output.__init__(self)
   def _plotCore(self, data):
      labels = data.getFilenames()

      F = data.getNumFiles()

      data.setAxis("none")
      data.setIndex(0)
      data.setFileIndex(0)
      clim = np.zeros(len(self._thresholds), 'float')
      for f in range(0, F):
         x = self._thresholds
         y = np.zeros([len(self._thresholds)], 'float')
         for t in range(0,len(self._thresholds)):
            threshold = self._thresholds[t]
            data.setFileIndex(f)
            data.setAxis("none")
            data.setIndex(0)
            var = data.getPvar(threshold)
            [obs, p] = data.getScores(["obs", var])

            color = self._getColor(f, F)
            style = self._getStyle(f, F)

            if(self._binType == "below"):
               p = p
               obs = obs < threshold
            elif(self._binType == "above"):
               p = 1 - p
               obs = obs > threshold
            else:
               Common.error("Bin type must be one of 'below' or 'above' for reliability plot")

            clim[t] = np.mean(obs)
            y[t] = np.mean(p)

         label = labels[f]
         mpl.plot(x, y, style, color=color, lw=self._lw, ms=self._ms, label=label)
      self._plotObs(x, clim)

      mpl.ylim([0,1])
      mpl.xlabel(data.getAxisLabel("threshold"))
      mpl.ylabel("Marginal probability")
      mpl.grid()

class Freq(Output):
   _description = "Show frequency of obs and forecasts"
   _reqThreshold = True
   _supX = False
   _experimental = True
   def __init__(self):
      Output.__init__(self)
   def _plotCore(self, data):
      labels = data.getFilenames()

      F = data.getNumFiles()

      data.setAxis("none")
      data.setIndex(0)
      data.setFileIndex(0)

      for f in range(0, F):
         # Setup x and y: When -b within, we need one less value in the array
         N = len(self._thresholds)
         x = self._thresholds
         if(self._binType == "within"):
            N = len(self._thresholds) - 1
            x = (self._thresholds[1:]+self._thresholds[:-1])/2
         y = np.zeros(N, 'float')
         clim = np.zeros(N, 'float')
         for t in range(0,N):
            threshold = self._thresholds[t]
            data.setFileIndex(f)
            data.setAxis("none")
            data.setIndex(0)
            [obs, fcst] = data.getScores(["obs", "fcst"])

            color = self._getColor(f, F)
            style = self._getStyle(f, F)

            if(self._binType == "below"):
               fcst = fcst < threshold
               obs = obs < threshold
            elif(self._binType == "above"):
               fcst = fcst > threshold
               obs = obs > threshold
            elif(self._binType == "within"):
               fcst = (fcst >= threshold) & (fcst < self._thresholds[t+1])
               obs = (obs >= threshold) & (obs < self._thresholds[t+1])

            clim[t] = np.mean(obs)
            y[t] = np.mean(fcst)

         label = labels[f]
         mpl.plot(x, y, style, color=color, lw=self._lw, ms=self._ms, label=label)
      self._plotObs(x, clim)

      mpl.ylim([0,1])
      mpl.xlabel(data.getAxisLabel("threshold"))
      mpl.ylabel("Frequency " + self._binType)
      mpl.grid()


class InvReliability(Output):
   _description = "Reliability diagram for a certain quantile (-r)"
   _reqThreshold = True
   _supX = False
   _experimental = True
   def __init__(self):
      Output.__init__(self)
   def _plotCore(self, data):
      labels = data.getFilenames()

      F = data.getNumFiles()
      ax  = mpl.gca()
      quantiles = self._thresholds
      if(quantiles[0] < 0.5):
         axi = mpl.axes([0.66,0.65,0.2,0.2])
      else:
         axi = mpl.axes([0.66,0.15,0.2,0.2])
      mpl.sca(ax)

      data.setAxis("none")
      data.setIndex(0)
      data.setFileIndex(0)
      for t in range(0,len(quantiles)):
         quantile = self._thresholds[t]
         var = data.getQvar(quantile)
         [obs, p] = data.getScores(["obs", var])

         # Determine the number of bins to use # (at least 11, at most 25)
         N = min(25, max(11, int(len(obs)/1000)))
         N = 21
         edges = np.linspace(0,20,N+1)
         #edges = [0,0.001,1,2,3,4,5,6,7,8,9,10]
         if(data.getVariable() == "Precip"):
            edges = np.linspace(0,np.sqrt(Common.nanmax(obs)), N+1)**2
         else:
            edges = np.linspace(Common.nanmin(obs),Common.nanmax(obs), N+1)

         #edges = np.zeros(N, 'float')
         #perc = np.linspace(0,100,N)
         #for i in range(0,N):
         #   edges[i] = Common.nanpercentile(obs, perc[i])
         #edges = np.unique(edges)

         x  = np.zeros([len(edges)-1,F], 'float')

         y = np.nan*np.zeros([F,len(edges)-1],'float')
         n = np.zeros([F,len(edges)-1],'float')
         v = np.zeros([F,len(edges)-1],'float')
         # Draw reliability lines
         for f in range(0, F):
            color = self._getColor(f, F)
            style = self._getStyle(f, F)
            data.setFileIndex(f)
            data.setAxis("none")
            data.setIndex(0)
            var = data.getQvar(quantile)
            [obs, p] = data.getScores(["obs", var])

            obs = obs <= p

            # Compute frequencies
            for i in range(0,len(edges)-1):
               q = (p >= edges[i])& (p < edges[i+1])
               I = np.where(q)[0]
               if(len(I) > 0):
                  n[f,i] = len(obs[I])
                  # Need at least 10 data points to be valid
                  if(n[f,i] >= 2):
                     y[f,i] = np.mean(obs[I])
                     v[f,i] = np.var(obs[I])
                  x[i,f] = np.mean(p[I])

            label = labels[f]
            if(not t == 0):
               label = ""
            mpl.plot(x[:,f], y[f], style, color=color, lw=self._lw, ms=self._ms, label=label)
         self._plotObs(edges,0*edges + quantile)

         # Draw confidence bands (do this separately so that these lines don't sneak into the legend)
         for f in range(0, F):
            color = self._getColor(f, F)
            self._plotConfidence(x[:,f], y[f], v[f], n[f], color=color)
            axi.plot(x[:,f], n[f], style, color=color, lw=self._lw, ms=self._ms)
            axi.xaxis.set_major_locator(mpl.NullLocator())
            axi.set_yscale('log')
            axi.set_title("Number")
      mpl.sca(ax)
      mpl.ylim([0,1])
      color = "gray"
      mpl.xlabel(data.getVariableAndUnits())
      mpl.ylabel("Observed frequency")
      units = " " + data.getUnits()
      mpl.title("Quantile: " + str(quantile*100) + "%")
