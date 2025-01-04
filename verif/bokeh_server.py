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
                print("Cannot parse %s" % filename)

        if len(self.inputs) == 0:
            verif.util.error("No valid files")

        self.init_data()

        self.times = None
        self.xlog = False
        self.ylog = False
        self.simple = True
        self.show_perfect = False
        self.legfs = 10
        self.use_mpl = use_mpl

        self.valid_axes = [x[0].lower() for x in verif.axis.get_all()]
        axes = [
            "leadtime",
            "time",
            "leadtimeday",
            "week",
            "month",
            "year",
            "location",
            "lat",
            "lon",
            "elev",
            "no",
            "threshold",
        ]
        self.valid_axes = [(x, x.capitalize()) for x in axes]
        self.valid_aggregators = [
            (x.name(), x.name().capitalize())
            for x in verif.aggregator.get_all()
            if x.name != "Quantile"
        ]
        self.valid_thresholds = [0]  # self.data.thresholds

        self.valid_leadtimes = self.data.leadtimes
        self.aggregator = verif.aggregator.Mean()
        self.threshold = 0  # self.valid_thresholds[0]
        plots = [
            "obsfcst",
            "timeseries",
            "qq",
            "taylor",
            "performance",
            "cond",
            "reliability",
        ]
        # self.valid_metrics = ["mae", "rmse", "corr", "taylor", "performance", "cond", "reliability"]

        self.valid_metrics = list()
        for m in verif.metric.get_all_by_type(verif.metric_type.Deterministic()):
            if m[1].is_valid():
                self.valid_metrics += [(m[0].lower(), m[1].name)]

        # Widgets
        self.widgets = list()
        self.select_metric = bokeh.models.widgets.Select(
            title="Metric:", value="mae", options=self.valid_metrics
        )
        self.select_metric.on_change("value", self.update)
        self.widgets.append(self.select_metric)
        self.select_axis = bokeh.models.widgets.Select(
            title="Axis:", value="leadtime", options=self.valid_axes
        )
        self.select_axis.on_change("value", self.update)
        self.widgets.append(self.select_axis)

        # Aggregator
        self.select_aggregator = bokeh.models.widgets.Select(
            title="Aggregator:", value="mean", options=self.valid_aggregators
        )
        self.select_aggregator.on_change("value", self.update)
        self.widgets.append(self.select_aggregator)

        # Time aggregator
        self.select_time_aggregator = bokeh.models.widgets.Select(
            title="Time aggregator:", value="mean", options=self.valid_aggregators
        )
        self.select_time_aggregator.on_change("value", self.update_data0)
        self.widgets.append(self.select_time_aggregator)

        # TODO: Don't hard code timescales
        time_aggregation_leadtimes = ["None"] + [f"{int(i)}" for i in self.valid_leadtimes if i > 0]
        self.select_time_aggregation = bokeh.models.widgets.Select(
                title="Time aggregation:", value="None", options=time_aggregation_leadtimes
        )
        self.select_time_aggregation.on_change("value", self.update_data0)
        self.widgets.append(self.select_time_aggregation)

        self.select_threshold = bokeh.models.widgets.Select(
            title="Threshold:",
            value="0",
            options=["%s" % s for s in self.valid_thresholds],
        )
        print(self.valid_thresholds)
        self.select_threshold.on_change("value", self.update)
        self.widgets.append(self.select_threshold)

        self.select_elevs = bokeh.models.widgets.Slider(
            title="Altitudes:",
            value=0,
            start=0,
            step=10,
            end=1000,
            # callback_policy="mouseup",
        )  # , callback_throttle=1e6)
        self.select_elevs.on_change("value", self.update)
        self.widgets.append(self.select_elevs)


        start_date = verif.util.unixtime_to_datetime(self.data.times[0])
        end_date = verif.util.unixtime_to_datetime(self.data.times[-1])
        self.select_leadtimes = bokeh.models.widgets.MultiSelect(
            title="Leadtimes:", options=[(str(x), str(x)) for x in self.data.leadtimes]
        )
        self.select_leadtimes.on_change("value", self.select_leadtimes_callback)
        self.widgets.append(self.select_leadtimes)

        self.checkbox_group = bokeh.models.widgets.CheckboxButtonGroup(
            labels=["show perfect", "simple", "legend"], active=[1, 2]
        )
        # self.checkbox_group.on_click(self.update0)
        self.widgets.append(self.checkbox_group)

        self.fullpanel = column(self.widgets)
        # tab1 = bokeh.models.widgets.Panel(child=column(self.widgets), title="Deterministic")
        # tab2 = bokeh.models.widgets.Panel(child=column(self.widgets), title="Probabilistic")
        # self.panel = bokeh.models.widgets.Tabs(tabs=[tab1, tab2])


        location_filter_widgets = self.widgets
        location_filter_panel = column(location_filter_widgets)

        self.panel = column(self.widgets + self.widgets)
        # self.panel = column([self.panel, location_filter_panel])

        self.figure = figure(toolbar_location="below", sizing_mode="stretch_both")
        self.layout = row(self.panel, self.figure, sizing_mode="stretch_both")

        self.create_figure()

    def get_panel_children(self, plot, data):
        children = list()
        children.append(bokeh.layouts.WidgetBox(self.select_metric))
        if self.plot.supports_x:
            children.append(bokeh.layouts.WidgetBox(self.select_axis))
        if self.plot.supports_threshold:
            pass
        print("Adding")
        children.append(bokeh.layouts.WidgetBox(self.select_threshold))
        children.append(bokeh.layouts.WidgetBox(self.select_elevs))
        children.append(bokeh.layouts.WidgetBox(self.checkbox_group))
        return children

    def init_data(self):
        self.data = verif.data.Data(self.inputs)

    def update_data0(self, attr, old, new):
        self.update_data()
        self.create_figure()

    def update_data(self):
        """
        Call this if options related to data have been changed
        """
        print("Updating data")
        altitudes = [self.select_elevs.value, 10000]
        time_aggregation = self.select_time_aggregation.value
        if time_aggregation == "None":
            time_aggregation = None
        else:
            time_aggregation = int(time_aggregation)
        time_aggregator = verif.aggregator.get(self.select_time_aggregator.value)

        leadtimes = [float(s) for s in self.select_leadtimes.value]
        if len(leadtimes) == 0:
            leadtimes = self.valid_leadtimes
        if isinstance(time_aggregation, int):
            leadtimes = [l for l in leadtimes if l >= time_aggregation]

        self.data = verif.data.Data(self.inputs,
                times=self.times,
                leadtimes=leadtimes,
                dim_agg_length=time_aggregation,
                dim_agg_method=time_aggregator,
                elev_range=altitudes)

    def create_figure(self):
        print("Create figure")
        # create a plot and style its properties
        self.figure = figure(toolbar_location="below")
        type = "linear"
        if self.axis.is_time_like:
            type = "datetime"

        # Set up metric
        metric_name = self.select_metric.value
        metric = None
        if metric_name in [x[0].lower() for x in verif.metric.get_all()]:
            metric = verif.metric.get(metric_name)
            aggregator = verif.aggregator.get(self.select_aggregator.value)
            metric.aggregator = aggregator
            self.plot = verif.output.Standard(metric)
        else:
            self.plot = verif.output.get(metric_name)
        assert self.plot is not None

        # Set up axis
        axis_name = self.select_axis.value
        axis = verif.axis.get(axis_name)
        self.plot.axis = axis

        # Set up threshold
        self.plot.thresholds = [
            float(self.select_threshold.value)
        ]  # np.linspace(0,10,11)#[3]
        self.plot.aggregator = self.aggregator
        self.plot.default_colors = ["red", "blue", "orange"]
        self.plot.figsize = [12, 8]
        self.plot.simple = 1 in self.checkbox_group.active
        self.plot.show_perfect = 0 in self.checkbox_group.active
        self.plot.legfs = (2 in self.checkbox_group.active) * 10
        self.select_axis.disabled = metric_name == "rmse"

        type = verif.driver.get_type(self.plot, metric)
        if type is not None:
            self.plot.thresholds = verif.driver.get_thresholds(type, self.data)
        if self.plot.require_threshold_type == "deterministic":
            self.plot.thresholds = np.linspace(0,10,11)
            # pass
        elif self.plot.require_threshold_type == "threshold" or (
            metric is not None and metric.require_threshold_type == "threshold"
        ):
            self.select_threshold.options = ["%g" % x for x in self.data.thresholds]
            # if axis_name == "threshold":
            #   self.plot.thresholds = self.data.thresholds

        # self.plot.xlog = self.xlog
        # self.plot.ylog = self.ylog
        # if self.axis == verif.axis.Location():
        #   self.plot.map(self.data)
        # else:
        if self.use_mpl:
            import bokeh.mpl

            self.plot.plot(self.data)
            q = bokeh.mpl.to_bokeh()
            print("Done")
            self.figure = q
        else:
            self.plot.bokeh(self.data, self.figure)

        # Create control panel
        curr = self.fullpanel.children[:]
        if (
            len(self.data.thresholds) == 0
            or (metric is not None and not metric.supports_threshold)
            or axis_name == "threshold"
        ):
            pass
            # curr.pop(3)
        if not self.plot.supports_x:
            curr.pop(1)
        # self.panel.children = self.get_panel_children(self.plot, self.data)
        self.panel.children = curr

        if metric_name == "rmse":
            # self.widgets[1].options = ["1", "2"]
            self.widgets[1].disabled = True

        self.layout.children[1] = self.figure

    def select_times_callback(self, attr, old, new):
        start_date = verif.util.datetime_to_date(new[0])
        end_date = verif.util.datetime_to_date(new[1])
        dates = verif.util.parse_numbers("%d:%d" % (start_date, end_date), True)
        self.times = [verif.util.date_to_unixtime(date) for date in dates]
        self.update_data()
        self.create_figure()

    def select_leadtimes_callback(self, attr, old, new):
        print(attr, old, new)
        self.update_data()
        self.create_figure()

    def update(self, attr, old, new):
        print("UPDATING", attr, old, new)
        self.create_figure()

    def update0(self, value):
        self.create_figure()


def modify_doc(doc):
    bokeh_app = verif.bokeh_server.BokehServer(filenames, use_mpl)
    doc.add_root(bokeh_app.layout)


def main():
    print("Opening Bokeh application on http://localhost:5006/")

    global filenames
    global use_mpl

    io_loop = IOLoop.current()
    filenames = [arg for arg in sys.argv[1:] if arg not in ["--mpl"]]
    use_mpl = "--mpl" in sys.argv

    bokeh_app = Application(FunctionHandler(modify_doc))
    server = Server(
        {"/": bokeh_app},
        io_loop=io_loop,
        allow_websocket_origin=["pc4423.pc.met.no:5006", "localhost:5006"],
    )
    server.start()

    # io_loop.add_callback(server.show, "/")
    io_loop.start()


if __name__ == "__main__":
    main()
