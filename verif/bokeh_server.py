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
   def __init__(self, filenames):
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
      self.simple = False
      self.show_perfect = False

      self.valid_axes = [x[0].lower() for x in verif.axis.get_all()]
      self.valid_axes = ["leadtime",
            "time", "leadtimeday", "week", "month", "year",
            "location", "lat", "lon", "elev",
            "no"]
      self.valid_aggregators = [x.name() for x in verif.aggregator.get_all() if x.name != "Quantile"]
      self.valid_thresholds = self.data.thresholds
      self.aggregator = verif.aggregator.Mean()
      self.threshold = self.valid_thresholds[0]
      self.valid_metrics = ["mae", "rmse", "corr", "taylor", "performance", "cond", "reliability"]
      p_fig = figure(toolbar_location=None)
      self.layout = bokeh.layouts.Column(children=[p_fig])
      self.create_figure()

   def update_data(self):
      """
      Call this if options related to data have been changed
      """
      print "Updating data"
      self.data = verif.data.Data(self.inputs, times=self.times,
            elev_range=self.altitudes)

   def update(self, doc):
      self.create_figure()

      control_panel = self.create_control_panel()
      self.ddoc = row(control_panel, self.layout)
      doc.add_root(self.ddoc)

   def create_control_panel(self):
      # Create the GUI
      select_metric = bokeh.models.widgets.Select(title="Metric:", value="mae", options=self.valid_metrics)
      select_metric.on_change("value", self.select_metric_callback)

      select_axis = bokeh.models.widgets.Select(title="Axis:", value="leadtime", options=self.valid_axes)
      select_axis.on_change("value", self.select_axis_callback)

      select_aggregator = bokeh.models.widgets.Select(title="Aggregator:", value="mean", options=self.valid_aggregators)
      select_aggregator.on_change("value", self.select_aggregator_callback)

      select_threshold = bokeh.models.widgets.Select(title="Threshold:", options=["%s" % s for s in self.valid_thresholds])
      select_threshold.on_change("value", self.select_threshold_callback)

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

      checkbox_group = bokeh.models.widgets.CheckboxButtonGroup( labels=["show perfect", "simple"])
      checkbox_group.on_click(self.select_log_axis_callback)

      button = bokeh.models.widgets.Button(label="<", button_type="success")
      button.on_click(self.hide_control_panel)

      #control_panel = column(button, select_metric, select_axis, select_times)
      return column(select_metric, select_axis, select_threshold, select_elevs, checkbox_group)

   def create_figure(self):
      # create a plot and style its properties
      type = "linear"
      if self.axis.is_time_like:
         type = "datetime"

      # plot = verif.output.Standard(metrics[i % len(metrics)])
      if self.metric in [x[0].lower() for x in verif.metric.get_all()]:
         plot = verif.output.Standard(verif.metric.get(self.metric))
      else:
         plot = verif.output.get(self.metric)
      plot.axis = self.axis
      plot.thresholds = [self.threshold] # np.linspace(0,10,11)#[3]
      plot.aggregator = self.aggregator
      plot.default_colors=["red","blue"]
      plot.figsize = [12,12]
      plot.simple = self.simple
      plot.show_perfect = self.show_perfect
      #plot.xlog = self.xlog
      #plot.ylog = self.ylog
      #if self.axis == verif.axis.Location():
      #   plot.map(self.data)
      #else:
      plot.plot(self.data)
      self.p_fig = bokeh.mpl.to_bokeh()
      #self.layout = bokeh.layouts.Column(children=[self.p_fig])
      self.layout.children = [self.p_fig]

   def select_metric_callback(self, attr, old, new):
      self.metric = new
      self.create_figure()

   def select_axis_callback(self, attr, old, new):
      self.axis = verif.axis.get(new)
      self.create_figure()

   def select_log_axis_callback(self, attr):
      self.show_perfect = 0 in attr
      self.simple = 1 in attr
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
