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
            "no", "threshold"]
      self.valid_axes = [(x, x.capitalize()) for x in axes]
      self.valid_aggregators = [(x.name(), x.name().capitalize()) for x in verif.aggregator.get_all() if x.name != "Quantile"]
      self.valid_thresholds = [0] # self.data.thresholds
      self.aggregator = verif.aggregator.Mean()
      self.threshold = 0#self.valid_thresholds[0]
      metrics = ["mae", "rmse", "corr", "error", "ets"]
      plots = ["obsfcst","taylor", "performance", "cond", "reliability"]
      # self.valid_metrics = ["mae", "rmse", "corr", "taylor", "performance", "cond", "reliability"]
      self.valid_metrics = [(x, verif.metric.get(x).description) for x in metrics]
      self.valid_metrics += [(x, x.capitalize() + " diagram") for x in plots]

      # Widgets
      self.widgets = list()
      self.select_metric = bokeh.models.widgets.Select(title="Metric:", value="mae", options=self.valid_metrics)
      self.select_metric.on_change("value", self.update)
      self.widgets.append(self.select_metric)
      self.select_axis = bokeh.models.widgets.Select(title="Axis:", value="leadtime", options=self.valid_axes)
      self.select_axis.on_change("value", self.update)
      self.widgets.append(self.select_axis)

      self.select_aggregator = bokeh.models.widgets.Select(title="Aggregator:", value="mean", options=self.valid_aggregators)
      self.select_aggregator.on_change("value", self.update)
      self.widgets.append(self.select_aggregator)

      self.select_threshold = bokeh.models.widgets.Select(title="Threshold:",
            value="0", options=["%s" % s for s in self.valid_thresholds])
      self.select_threshold.on_change("value", self.update)
      self.widgets.append(self.select_threshold)

      self.select_elevs = bokeh.models.widgets.Slider(title="Altitudes:", value=0, start=0, step=10, end=1000, callback_policy="mouseup") # , callback_throttle=1e6)
      self.select_elevs.on_change("value", self.update)
      self.widgets.append(self.select_elevs)
      self.update_data()

      start_date = verif.util.unixtime_to_datetime(self.data.times[0])
      end_date = verif.util.unixtime_to_datetime(self.data.times[-1])
      self.select_times = bokeh.models.widgets.MultiSelect(title="Leadtimes:", options=[(str(x),str(x)) for x in self.data.leadtimes])

      self.checkbox_group = bokeh.models.widgets.CheckboxButtonGroup( labels=["show perfect", "simple", "legend"], active=[1,2])
      self.checkbox_group.on_click(self.update0)
      self.widgets.append(self.checkbox_group)

      self.fullpanel = column(self.widgets)
      self.panel = column(self.widgets)

      self.figure = figure(toolbar_location="below")
      self.layout = row(self.panel, self.figure)

      self.create_figure()

   def get_panel_children(self, plot, data):
      children = list()
      children.append(bokeh.layouts.WidgetBox(self.select_metric))
      if self.plot.supports_x:
         children.append(bokeh.layouts.WidgetBox(self.select_axis))
      if self.plot.supports_threshold:
         children.append(bokeh.layouts.WidgetBox(self.select_threshold))
      children.append(bokeh.layouts.WidgetBox(self.select_elevs))
      children.append(bokeh.layouts.WidgetBox(self.checkbox_group))
      return children

   def update_data(self):
      """
      Call this if options related to data have been changed
      """
      print "Updating data"
      altitudes = [self.select_elevs.value, 10000]
      self.data = verif.data.Data(self.inputs, times=self.times, elev_range=altitudes)

   def modify_doc(self, doc):
      doc.add_root(self.layout)

   def create_figure(self):
      print "Create figure"
      # create a plot and style its properties
      self.figure = figure(toolbar_location="below")
      type = "linear"
      if self.axis.is_time_like:
         type = "datetime"

      # Set up metric
      metric_name = self.select_metric.value
      if metric_name in [x[0].lower() for x in verif.metric.get_all()]:
         metric = verif.metric.get(metric_name)
         metric.aggregator = self.aggregator
         self.plot = verif.output.Standard(metric)
      else:
         self.plot = verif.output.get(metric_name)
      assert(self.plot is not None)

      # Set up axis
      axis = verif.axis.get(self.select_axis.value)
      self.plot.axis = axis

      # Set up threshold
      self.plot.thresholds = [float(self.select_threshold.value)] # np.linspace(0,10,11)#[3]
      self.plot.aggregator = self.aggregator
      self.plot.default_colors=["red","blue","orange"]
      self.plot.figsize = [12,8]
      self.plot.simple = 1 in self.checkbox_group.active
      self.plot.show_perfect = 0 in self.checkbox_group.active
      self.plot.legfs = (2 in self.checkbox_group.active) * 10
      self.select_axis.disabled = metric_name == "rmse"

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

      # Create control panel
      curr = self.fullpanel.children[:]
      print self.data.thresholds
      if len(self.data.thresholds) == 0:
         curr.pop(3)
      if not self.plot.supports_x:
         curr.pop(1)
      #self.panel.children = self.get_panel_children(self.plot, self.data)
      self.panel.children = curr

      if metric_name == "rmse":
         #self.widgets[1].options = ["1", "2"]
         self.widgets[1].disabled = True

      self.layout.children[1] = self.figure

   def select_times_callback(self, attr, old, new):
      start_date = verif.util.datetime_to_date(new[0])
      end_date = verif.util.datetime_to_date(new[1])
      dates = verif.util.parse_numbers("%d:%d" % (start_date, end_date), True)
      self.times = [verif.util.date_to_unixtime(date) for date in dates]
      self.update_data()
      self.create_figure()

   def update(self, attr, old, new):
      self.create_figure()

   def update0(self, value):
      self.create_figure()


def main():
   print('Opening Bokeh application on http://localhost:5006/')

   io_loop = IOLoop.current()
   filenames = [arg for arg in sys.argv[1:] if arg not in ["--mpl"]]
   use_mpl = "--mpl" in sys.argv
   s = verif.bokeh_server.BokehServer(filenames, use_mpl)

   bokeh_app = Application(FunctionHandler(s.modify_doc))
   server = Server({'/': bokeh_app}, io_loop=io_loop,
         allow_websocket_origin=["pc4423.pc.met.no:5006", "localhost:5006"])
   server.start()

   #io_loop.add_callback(server.show, "/")
   io_loop.start()


if __name__ == '__main__':
   main()
