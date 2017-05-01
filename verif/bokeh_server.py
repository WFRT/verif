import os
import sys
import numpy as np

from bokeh.layouts import column, row
from bokeh.server.server import Server
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from tornado.ioloop import IOLoop
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
import bokeh.plotting
import bokeh.mpl
import verif.data
import verif.output
import verif.input
import verif.metric
import verif.axis
import verif.aggregator
import datetime


def datetime_to_date(datetime):
   x = datetime.year * 10000 + datetime.month * 100 + datetime.day
   return x


def unixtime_to_datetime(unixtime):
   x = verif.util.unixtime_to_datenum(unixtime)
   x = datetime.datetime.fromordinal(int(x))
   return x


def datetime_to_date(datetime):
   x = datetime.year * 10000 + datetime.month * 100 + datetime.day
   return x


class BokehServer(object):
   def __init__(self, filenames, use_mpl=False):
      self.axis = verif.axis.Leadtime()
      self.metric = "mae"
      self.inputs = list()
      for filename in filenames:
         try:
            input = verif.input.get_input(filename)
            self.inputs += [input]
         except:
            print "Cannot parse %s" % filename

      if len(self.inputs) == 0:
         verif.util.error("No valid files")
      self.times = None
      self.altitudes = None
      self.update_data()
      self.xlog = False
      self.ylog = False
      self.simple = True
      self.show_perfect = False
      self.legfs = 10
      self.use_mpl = use_mpl

      self.valid_axes = [x[0].lower() for x in verif.axis.get_all()]
      axes = ["leadtime",
            "time", "leadtimeday", "week", "month", "year",
            "location", "lat", "lon", "elev",
            "no"]
      self.valid_axes = [(x, x.capitalize()) for x in axes]
      self.valid_aggregators = [(x.name(), x.name().capitalize()) for x in verif.aggregator.get_all() if x.name != "Quantile"]
      self.valid_thresholds = [0] # self.data.thresholds
      self.aggregator = verif.aggregator.Mean()
      self.threshold = 0#self.valid_thresholds[0]
      metrics = ["mae", "rmse", "corr", "error"]
      plots = ["obsfcst","taylor", "performance", "cond", "reliability"]
      # self.valid_metrics = ["mae", "rmse", "corr", "taylor", "performance", "cond", "reliability"]
      self.valid_metrics = [(x, verif.metric.get(x).description) for x in metrics]
      self.valid_metrics += [(x, x.capitalize() + " diagram") for x in plots]
      self.figure = figure(toolbar_location="below")
      self.control_panel = bokeh.layouts.Column(children=[])
      self.layout = row(self.control_panel, self.figure)
      self.create_figure()

   def update_data(self):
      """
      Call this if options related to data have been changed
      """
      print "Updating data"
      self.data = verif.data.Data(self.inputs, times=self.times,
            elev_range=self.altitudes)

   def modify_doc(self, doc):
      doc.add_root(self.layout)

   def update_control_panel(self):
      # Create the GUI
      widgets = list()
      select_metric = bokeh.models.widgets.Select(title="Metric:", options=self.valid_metrics)
      select_metric.on_change("value", self.select_metric_callback)
      widgets.append(select_metric)

      if self.plot.supports_x:
         select_axis = bokeh.models.widgets.Select(title="Axis:", value="leadtime", options=self.valid_axes)
         select_axis.on_change("value", self.select_axis_callback)
         widgets.append(select_axis)

      #if self.plot._metric.supports_aggregator:
      select_aggregator = bokeh.models.widgets.Select(title="Aggregator:", value="mean", options=self.valid_aggregators)
      select_aggregator.on_change("value", self.select_aggregator_callback)
      widgets.append(select_aggregator)

      select_threshold = bokeh.models.widgets.Select(title="Threshold:", options=["%s" % s for s in self.valid_thresholds])
      select_threshold.on_change("value", self.select_threshold_callback)
      widgets.append(select_threshold)

      start_date = unixtime_to_datetime(self.data.times[0])
      end_date = unixtime_to_datetime(self.data.times[-1])
      #select_times = bokeh.models.widgets.DateRangeSlider(title="End date:")#, bounds=[start_date,end_date])
      #select_times = bokeh.models.widgets.MultiSelect(title="Stations:", options=[(str(x.id),str(x.id)) for x in data.locations])
      select_times = bokeh.models.widgets.MultiSelect(title="Leadtimes:", options=[(str(x),str(x)) for x in self.data.leadtimes])
      #select_times.on_change("range", select_times_callback)
      #select_elevs = bokeh.models.widgets.RangeSlider(title="Altitudes:",
      #      range=[0,1000], start=0, step=10, end=1000, callback_policy="mouseup",
      #      callback_throttle=1e6)
      select_elevs = bokeh.models.widgets.Slider(title="Altitudes:",
            start=0, step=10, end=1000, callback_policy="mouseup") # , callback_throttle=1e6)
      select_elevs.on_change("value", self.select_altitude_callback)
      widgets.append(select_elevs)

      checkbox_group = bokeh.models.widgets.CheckboxButtonGroup( labels=["show perfect", "simple", "legend"], active=[1,2])
      checkbox_group.on_click(self.select_log_axis_callback)
      widgets.append(checkbox_group)

      button = bokeh.models.widgets.Button(label="<", button_type="success")
      button.on_click(self.hide_control_panel)
      self.layout.children[0] = column(widgets)

   def create_figure(self):
      # create a plot and style its properties
      self.figure = figure(toolbar_location="below")
      type = "linear"
      if self.axis.is_time_like:
         type = "datetime"

      # self.plot = verif.output.Standard(metrics[i % len(metrics)])
      if self.metric in [x[0].lower() for x in verif.metric.get_all()]:
         metric = verif.metric.get(self.metric)
         metric.aggregator = self.aggregator
         self.plot = verif.output.Standard(metric)
      else:
         self.plot = verif.output.get(self.metric)
      #button = bokeh.models.widgets.Button(label="<", button_type="success")
      #self.control_panel.children = [button]
      #print len(self.control_panel.children)
      self.plot.axis = self.axis
      self.plot.thresholds = [self.threshold] # np.linspace(0,10,11)#[3]
      self.plot.aggregator = self.aggregator
      self.plot.default_colors=["red","blue","orange"]
      self.plot.figsize = [12,8]
      self.plot.simple = self.simple
      self.plot.show_perfect = self.show_perfect
      self.plot.legfs = self.legfs
      #self.plot.xlog = self.xlog
      #self.plot.ylog = self.ylog
      #if self.axis == verif.axis.Location():
      #   self.plot.map(self.data)
      #else:
      if self.use_mpl:
         self.plot.plot(self.data)
         self.figure = bokeh.mpl.to_bokeh()
      else:
         self.plot.bokeh(self.data, self.figure)
      self.layout.children[1] = self.figure
      self.update_control_panel()

   def select_metric_callback(self, attr, old, new):
      self.metric = new
      self.create_figure()

   def select_axis_callback(self, attr, old, new):
      self.axis = verif.axis.get(new)
      self.create_figure()

   def select_log_axis_callback(self, attr):
      self.show_perfect = 0 in attr
      self.simple = 1 in attr
      self.legfs = (2 in attr) * 10
      if 2 in attr:
         self.legfs = 10
      else:
         self.legfs = 0
      self.create_figure()

   def select_altitude_callback(self, attr, old, new):
      self.altitudes = [new, 10000]
      self.update_data()
      self.create_figure()

   def select_aggregator_callback(self, attr, old, new):
      self.aggregator = verif.aggregator.get(new)
      self.create_figure()

   def select_threshold_callback(self, attr, old, new):
      self.threshold = float(new)
      self.create_figure()

   def select_times_callback(self, attr, old, new):
      start_date = datetime_to_date(new[0])
      end_date = datetime_to_date(new[1])
      dates = verif.util.parse_numbers("%d:%d" % (start_date, end_date), True)
      self.times = [verif.util.date_to_unixtime(date) for date in dates]
      self.update_data()
      self.create_figure()

   def hide_control_panel(self):
      doc.children = [column(button), self.layout]
      self.create_figure()
