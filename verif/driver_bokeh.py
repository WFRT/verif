import os
import sys

from bokeh.layouts import column, row
from bokeh.server.server import Server
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from tornado.ioloop import IOLoop
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
import bokeh.plotting
import verif.data
import verif.output
import verif.input
import verif.metric
import verif.axis
import verif.aggregator
import datetime

def modify_doc(doc):
   # create a plot and style its properties
   p_fig = figure(toolbar_location=None)
   p_fig.outline_line_color = None
   p_fig.grid.grid_line_color = None

   i = 0

   #filenames = [filename for filename in os.listdir("./") if os.path.isfile(filename)]
   if len(sys.argv) == 1:
      verif.util.error("Must specify at least one valid verif file")
   filenames = sys.argv[1:]
   inputs = list()
   for filename in filenames:
      try:
         input = verif.input.get_input(filename)
         inputs += [input]
      except:
         print "Cannot parse %s" % filename

   if len(inputs) == 0:
      verif.util.error("No valid files")
   data = verif.data.Data(inputs)
   metric = verif.metric.Mae()
   metrics = [verif.metric.Mae(), verif.metric.Rmse(), verif.metric.Corr()]
   from bokeh.models.layouts import VBox
   layout = VBox(children=[p_fig])

   i = 0
   valid_axes = [x[0].lower() for x in verif.axis.get_all()]
   valid_axes = ["leadtime", "time", "week", "month", "year", "location"]
   valid_aggregators = [x.name() for x in verif.aggregator.get_all() if x.name != "Quantile"]
   axis = verif.axis.Leadtime()
   global axis
   aggregator = verif.aggregator.Mean()
   global metric
   global aggregator
   global data

   def create_figure():
      global metric
      global axis
      global aggregator
      global data
      # create a plot and style its properties
      type = "linear"
      if axis.is_time_like:
         type = "datetime"
      p_fig = figure(toolbar_location=None, x_axis_type=type)
      #p_fig.border_fill_color = 'black'
      #p_fig.background_fill_color = 'black'
      p_fig.outline_line_color = None
      p_fig.grid.grid_line_color = None


      # plot = verif.output.Standard(metrics[i % len(metrics)])
      plot = verif.output.Standard(metric)
      plot.axis = axis
      plot.aggregator = aggregator
      plot.default_colors=["red","blue"]
      plot.plot(data, p_fig)
      layout.children = [p_fig]

   def select_metric_callback(attr, old, new):
      global metric
      metric = verif.metric.get(new)
      create_figure()

   def select_axis_callback(attr, old, new):
      global axis
      axis = verif.axis.get(new)
      create_figure()

   def select_aggregator_callback(attr, old, new):
      global aggregator
      aggregator = verif.aggregator.get(new)
      create_figure()

   def select_times_callback(attr, old, new):
      global aggregator
      global data
      start_date = datetime_to_date(new[0])
      end_date = datetime_to_date(new[1])
      print new
      dates = verif.util.parse_numbers("%d:%d" % (start_date, end_date), True)
      times = [verif.util.date_to_unixtime(date) for date in dates]
      data = verif.data.Data(inputs, times=times)
      create_figure()

   def hide_control_panel():
      doc.children = [column(button), layout]
      create_figure()

   def unixtime_to_datetime(unixtime):
      x = verif.util.unixtime_to_datenum(unixtime)
      x = datetime.datetime.fromordinal(int(x))
      return x

   def datetime_to_date(datetime):
      x = datetime.year * 10000 + datetime.month * 100 + datetime.day
      return x

   # add a button widget and configure with
   # the call back
   create_figure()
   select_metric = bokeh.models.widgets.Select(title="Metric:", value="mae", options=["mae", "rmse", "corr"])
   select_metric.on_change("value", select_metric_callback)

   select_axis = bokeh.models.widgets.Select(title="Axis:", value="leadtime", options=valid_axes)
   select_axis.on_change("value", select_axis_callback)

   select_aggregator = bokeh.models.widgets.Select(title="Aggregator:", value="mean", options=valid_aggregators)
   select_aggregator.on_change("value", select_aggregator_callback)

   start_date = unixtime_to_datetime(data.times[0])
   end_date = unixtime_to_datetime(data.times[-1])
   print start_date,end_date
   #select_times = bokeh.models.widgets.DateRangeSlider(title="End date:")#, bounds=[start_date,end_date])
   #select_times = bokeh.models.widgets.MultiSelect(title="Stations:", options=[(str(x.id),str(x.id)) for x in data.locations])
   select_times = bokeh.models.widgets.MultiSelect(title="Leadtimes:", options=[(str(x),str(x)) for x in data.leadtimes])
   #select_times.on_change("range", select_times_callback)

   button = bokeh.models.widgets.Button(label="<", button_type="success")
   button.on_click(hide_control_panel)

   # put the button and plot in a layout
   # and add to the document
   control_panel = column(button, select_metric, select_axis, select_times)
   ddoc = row(control_panel, layout)
   doc.add_root(ddoc)


def main():
   print('Opening Bokeh application on http://localhost:5006/')

   io_loop = IOLoop.current()

   bokeh_app = Application(FunctionHandler(modify_doc))
   server = Server({'/': bokeh_app}, io_loop=io_loop)
   server.start()

   io_loop.add_callback(server.show, "/")
   io_loop.start()


if __name__ == '__main__':
   main()
