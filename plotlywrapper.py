"""plotly wrapper to make easy plots easy to make"""

from __future__ import division

from tempfile import NamedTemporaryFile
from collections import defaultdict

# pylint: disable=redefined-builtin
from builtins import zip

import plotly.offline as py
import plotly.graph_objs as go

import numpy as np
import pandas as pd


__version__ = '0.0.29'


def _recursive_dict(*args):
    recursive_factory = lambda: defaultdict(recursive_factory)
    return defaultdict(recursive_factory, *args)


def _labels(base='trace'):
    i = 0
    while True:
        yield base + ' ' + str(i)
        i += 1


def _detect_notebook():
    """
    This isn't 100% correct but seems good enough

    Returns
    -------
    bool
        True if it detects this is a notebook, otherwise False.
    """
    try:
        from IPython import get_ipython
        from ipykernel import zmqshell
    except ImportError:
        return False
    kernel = get_ipython()
    try:
        from spyder.utils.ipython.spyder_kernel import SpyderKernel
        if isinstance(kernel.kernel, SpyderKernel):
            return False
    except (ImportError, AttributeError):
        pass
    return isinstance(kernel, zmqshell.ZMQInteractiveShell)


def _merge_layout(x, y):
    z = y.copy()
    if 'shapes' in z and 'shapes' in x:
        x['shapes'] += z['shapes']
    z.update(x)
    return z


def _try_pydatetime(x):
    """Opportunistically try to convert to pandas time indexes
    since plotly doesn't know how to handle them.
    """
    try:
        x = [y.isoformat() for y in x.to_pydatetime()]
    except AttributeError:
        pass
    return x


class Chart(object):
    """
    Plotly chart base class, usually this object will get created
    by from a function.
    """

    def __init__(self, data=None, layout=None, repr_plot=True):
        self.repr_plot = repr_plot
        self.data = data
        if data is None:
            self.data = []
        self.layout = layout
        if layout is None:
            layout = {}
        self.layout = _recursive_dict(layout)
        self.figure_ = None

    def __add__(self, other):
        self.data += other.data
        self.layout = _merge_layout(self.layout, other.layout)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def width(self, value):
        """Sets the width of the plot in pixels.

        Parameters
        ----------
        value : int
            Width of the plot in pixels.

        Returns
        -------
        Chart

        """
        self.layout['width'] = value
        return self

    def height(self, value):
        """Sets the height of the plot in pixels.

        Parameters
        ----------
        value : int
            Height of the plot in pixels.

        Returns
        -------
        Chart

        """
        self.layout['height'] = value
        return self

    def group(self):
        """Sets bar graph display mode to "grouped".

        Returns
        -------
        Chart

        """
        self.layout['barmode'] = 'group'
        return self

    def stack(self):
        """Sets bar graph display mode to "stacked".

        Returns
        -------
        Chart

        """
        self.layout['barmode'] = 'stack'
        return self

    def legend(self, visible=True):
        """Make legend visible.

        Parameters
        ----------
        visible : bool, optional

        Returns
        -------
        Chart

        """
        self.layout['showlegend'] = visible
        return self

    def xlabel(self, label):
        """Sets the x-axis title.

        Parameters
        ----------
        value : str
            Label for the x-axis

        Returns
        -------
        Chart

        """
        self.layout['xaxis']['title'] = label
        return self

    def ylabel(self, label, index=1):
        """Sets the y-axis title.

        Parameters
        ----------
        value : str
            Label for the y-axis
        index : int
            Y-axis index

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['title'] = label
        return self

    def zlabel(self, label):
        """Sets the z-axis title.

        Parameters
        ----------
        value : str
            Label for the z-axis

        Returns
        -------
        Chart

        """
        self.layout['zaxis']['title'] = label
        return self

    def xtickangle(self, angle):
        """Sets the angle of the x-axis tick labels.

        Parameters
        ----------
        value : int
            Angle in degrees

        Returns
        -------
        Chart

        """
        self.layout['xaxis']['tickangle'] = angle
        return self

    def ytickangle(self, angle, index=1):
        """Sets the angle of the y-axis tick labels.

        Parameters
        ----------
        value : int
            Angle in degrees
        index : int, optional
            Y-axis index

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickangle'] = angle
        return self

    def xlabelsize(self, size):
        """Set the size of the label

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['xaxis']['titlefont']['size'] = size
        return self

    def ylabelsize(self, size, index=1):
        """Set the size of the label

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['titlefont']['size'] = size
        return self

    def xticksize(self, size):
        """Set the tick font size

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['xaxis']['tickfont']['size'] = size
        return self

    def yticksize(self, size, index=1):
        """Set the tick font size

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickfont']['size'] = size
        return self

    def ytickvals(self, values, index=1):
        """Set the tick values

        Parameters
        ----------
        values : array-like

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickvals'] = values
        return self

    def yticktext(self, labels, index=1):
        """Set the tick labels

        Parameters
        ----------
        labels : array-like

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['ticktext'] = labels
        return self

    def xlim(self, low, high):
        """Set xaxis limits

        Parameters
        ----------
        low : number
        high : number

        Returns
        -------
        Chart

        """
        self.layout['xaxis']['range'] = [low, high]
        return self

    def ylim(self, low, high, index=1):
        """Set yaxis limits

        Parameters
        ----------
        low : number
        high : number
        index : int, optional

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['range'] = [low, high]
        return self

    def xdtick(self, dtick):
        self.layout['xaxis']['dtick'] = dtick
        return self

    def ydtick(self, dtick, index=1):
        self.layout['yaxis' + str(index)]['dtick'] = dtick
        return self

    def xnticks(self, nticks):
        self.layout['xaxis']['nticks'] = nticks
        return self

    def ynticks(self, nticks, index=1):
        self.layout['yaxis' + str(index)]['nticks'] = nticks
        return self

    def yaxis_left(self, index=1):
        """Puts the yaxis on the left hand side

        Parameters
        ----------
        index : int, optional

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['side'] = 'left'

    def yaxis_right(self, index=1):
        """Puts the yaxis on the right hand side

        Parameters
        ----------
        index : int, optional

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['side'] = 'right'

    def title(self, string):
        """Sets the title of the plot

        Parameters
        ----------
        string : str

        Returns
        -------
        Chart

        """
        self.layout['title'] = string
        return self

    def __repr__(self):
        if self.repr_plot:
            self.show(filename=None, auto_open=False)
        return super(Chart, self).__repr__()

    def show(self, filename=None, show_link=True, auto_open=True, detect_notebook=True):
        """Display the chart.

        Parameters
        ----------
        filename : str, optional
            Save plot to this filename, otherwise it's saved to a temporary file.
        show_link : bool, optional
            Show link to plotly.
        auto_open : bool, optional
            Automatically open the plot (in the browser).
        detect_notebook : bool, optional
            Try to detect if we're running in a notebook.

        """
        kargs = {}
        if detect_notebook and _detect_notebook():
            py.init_notebook_mode()
            plot = py.iplot
        else:
            plot = py.plot
            if filename is None:
                filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
            kargs['filename'] = filename
            kargs['auto_open'] = auto_open

        self.figure_ = go.Figure(data=self.data, layout=go.Layout(**self.layout))
        plot(self.figure_, show_link=show_link, **kargs)

    def save(self, filename=None, show_link=True, auto_open=False,
             output='file', plotlyjs=True):
        if filename is None:
            filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
        self.figure_ = go.Figure(data=self.data, layout=go.Layout(**self.layout))
        # NOTE: this doesn't work for output 'div'
        py.plot(self.figure_, show_link=show_link, filename=filename, auto_open=auto_open,
                output_type=output, include_plotlyjs=plotlyjs)
        return filename

    def to_json(self):
        """ Deprecated.
        """
        listdata = []
        for data in self.data:
            td = {}
            for k, v in data.items():
                try:
                    td[k] = v.tolist()
                except (AttributeError, TypeError):
                    td[k] = v
            listdata.append(td)
        return dict(data=listdata, layout=self.layout)

    @property
    def dict(self):
        return dict(data=self.data, layout=self.layout)


def spark_shape(points, shapes, fill=None, color='blue', width=5, yindex=0, heights=None):
    """TODO: Docstring for spark.

    Parameters
    ----------
    points : array-like
    shapes : array-like
    fill : array-like, optional

    Returns
    -------
    Chart

    """
    assert len(points) == len(shapes) + 1
    data = [{'marker': {'color': 'white'},
             'x': [points[0], points[-1]],
             'y': [yindex, yindex]}]

    if fill is None:
        fill = [False] * len(shapes)

    if heights is None:
        heights = [0.4] * len(shapes)

    lays = []
    for i, (shape, height) in enumerate(zip(shapes, heights)):
        if shape is None:
            continue
        if fill[i]:
            fillcolor = color
        else:
            fillcolor = 'white'
        lays.append(
            dict(type=shape,
                 x0=points[i], x1=points[i+1],
                 y0=yindex - height, y1=yindex + height,
                 xref='x', yref='y',
                 fillcolor=fillcolor,
                 line=dict(color=color,
                           width=width))
        )

    layout = dict(shapes=lays)

    return Chart(data=data, layout=layout)


def vertical(x, ymin=0, ymax=1, color=None, width=None, dash=None, opacity=None):
    """Draws a vertical line from `ymin` to `ymax`.

    Parameters
    ----------
    xmin : int, optional
    xmax : int, optional
    color : str, optional
    width : number, optional

    Returns
    -------
    Chart

    """
    lineattr = {}
    if color:
        lineattr['color'] = color
    if width:
        lineattr['width'] = width
    if dash:
        lineattr['dash'] = dash

    layout = dict(shapes=[dict(type='line',
                               x0=x, x1=x,
                               y0=ymin, y1=ymax,
                               opacity=opacity,
                               line=lineattr)])
    return Chart(layout=layout)


def horizontal(y, xmin=0, xmax=1, color=None, width=None, dash=None, opacity=None):
    """Draws a horizontal line from `xmin` to `xmax`.

    Parameters
    ----------
    xmin : int, optional
    xmax : int, optional
    color : str, optional
    width : number, optional

    Returns
    -------
    Chart

    """
    lineattr = {}
    if color:
        lineattr['color'] = color
    if width:
        lineattr['width'] = width
    if dash:
        lineattr['dash'] = dash

    layout = dict(shapes=[dict(type='line',
                               x0=xmin, x1=xmax,
                               y0=y, y1=y,
                               opacity=opacity,
                               line=lineattr)])
    return Chart(layout=layout)


def line(x=None, y=None, label=None, color=None, width=None, dash=None, opacity=None,
         mode='lines+markers', yaxis=1, fill=None, text="",
         markersize=6):
    """Draws connected dots.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    label : array-like, optional

    Returns
    -------
    Chart

    """
    assert x is not None or y is not None, "x or y must be something"
    yn = 'y' + str(yaxis)
    lineattr = {}
    if color:
        lineattr['color'] = color
    if width:
        lineattr['width'] = width
    if dash:
        lineattr['dash'] = dash
    if y is None:
        y = x
        x = None
    if x is None:
        x = np.arange(len(y))
    else:
        x = _try_pydatetime(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    assert x.shape[0] == y.shape[0]
    if y.ndim == 2:
        if not hasattr(label, '__iter__'):
            if label is None:
                label = _labels()
            else:
                label = _labels(label)
        data = [go.Scatter(x=x, y=yy, name=ll, line=lineattr, mode=mode, text=text,
                           fill=fill, opacity=opacity, yaxis=yn, marker=dict(size=markersize))
                for ll, yy in zip(label, y.T)]
    else:
        data = [go.Scatter(x=x, y=y, name=label, line=lineattr, mode=mode, text=text,
                           fill=fill, opacity=opacity, yaxis=yn, marker=dict(size=markersize))]
    if yaxis == 1:
        return Chart(data=data)

    return Chart(data=data, layout={'yaxis' + str(yaxis): dict(overlaying='y')})


def line3d(x, y, z, label=None, color=None, width=None, dash=None, opacity=None,
           mode='lines+markers'):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    assert x.shape == y.shape
    assert y.shape == z.shape
    lineattr = {}
    if color:
        lineattr['color'] = color
    if width:
        lineattr['width'] = width
    if dash:
        lineattr['dash'] = dash
    if y.ndim == 2:
        if not hasattr(label, '__iter__'):
            if label is None:
                label = _labels()
            else:
                label = _labels(label)
        data = [go.Scatter3d(x=xx, y=yy, z=zz, name=ll, line=lineattr, mode=mode,
                             opacity=opacity)
                for ll, xx, yy, zz in zip(label, x.T, y.T, z.T)]
    else:
        data = [go.Scatter3d(x=x, y=y, z=z, name=label, line=lineattr, mode=mode,
                             opacity=opacity)]
    return Chart(data=data)


def scatter3d(x, y, z, label=None, color=None, width=None, dash=None, opacity=None,
              mode='markers'):
    """3D Scatter Plot

    Parameters
    ----------
    x : array-like
        data on x-dimension
    y : array-like
        data on y-dimension
    z : array-like
        data on z-dimension
    label : TODO, optional
    mode : 'group' or 'stack', default 'group'
    opacity : TODO, optional

    Returns
    -------
    Chart

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    assert x.shape == y.shape
    assert y.shape == z.shape
    lineattr = {}
    if color:
        lineattr['color'] = color
    if width:
        lineattr['width'] = width
    if dash:
        lineattr['dash'] = dash
    if y.ndim == 2:
        if not hasattr(label, '__iter__'):
            if label is None:
                label = _labels()
            else:
                label = _labels(label)
        data = [go.Scatter3d(x=xx, y=yy, z=zz, name=ll, line=lineattr, mode=mode,
                             opacity=opacity)
                for ll, xx, yy, zz in zip(label, x.T, y.T, z.T)]
    else:
        data = [go.Scatter3d(x=x, y=y, z=z, name=label, line=lineattr, mode=mode,
                             opacity=opacity)]
    return Chart(data=data)


def scatter(x=None, y=None, label=None, color=None, width=None, dash=None, opacity=None,
            markersize=6, yaxis=1, fill=None, text="", mode='markers'):
    """Draws dots.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    label : array-like, optional

    Returns
    -------
    Chart

    """
    return line(x=x, y=y, label=label, color=color, width=width, dash=dash, opacity=opacity,
                mode=mode, yaxis=yaxis, fill=fill, text=text, markersize=markersize)


def bar(x=None, y=None, label=None, mode='group', yaxis=1, opacity=None):
    """Create a bar chart

    Parameters
    ----------
    x : array-like, optional
    y : TODO, optional
    label : TODO, optional
    mode : 'group' or 'stack', default 'group'
    opacity : TODO, optional

    Returns
    -------
    Chart
        A Chart with bar graph data.

    """
    assert x is not None or y is not None, "x or y must be something"
    yn = 'y' + str(yaxis)
    if y is None:
        y = x
        x = None
    if x is None:
        x = np.arange(len(y))
    else:
        x = _try_pydatetime(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if y.ndim == 2:
        if not hasattr(label, '__iter__'):
            if label is None:
                label = _labels()
            else:
                label = _labels(label)
        data = [go.Bar(x=x, y=yy, name=ll, yaxis=yn, opacity=opacity) for ll, yy in zip(label, y.T)]
    else:
        data = [go.Bar(x=x, y=y, name=label, yaxis=yn, opacity=opacity)]
    if yaxis == 1:
        return Chart(data=data, layout={'barmode': mode})

    return Chart(data=data, layout={'barmode': mode,
                                    'yaxis' + str(yaxis): dict(overlaying='y')})


def heatmap(z, x=None, y=None, colorscale='Viridis'):
    """Create a heatmap

    Parameters
    ----------
    z : TODO
    x : TODO, optional
    y : TODO, optional
    colorscale : TODO, optional

    Returns
    -------
    Chart


    """
    z = np.atleast_1d(z)
    data = [go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale
    )]
    return Chart(data=data)


def fill_zero(x=None, y=None, label=None, color=None, width=None, dash=None, opacity=None,
              mode='lines+markers', **kargs):
    """Fill to zero.

    Parameters
    ----------
    x : array-like, optional
    y : TODO, optional
    label : TODO, optional

    Returns
    -------
    Chart

    """
    return line(x=x, y=y, label=label, color=color, width=width, dash=dash,
                opacity=opacity, mode=mode, fill='tozeroy', **kargs)


def fill_between(x=None, ylow=None, yhigh=None, label=None, color=None, width=None, dash=None,
                 opacity=None, mode='lines+markers', **kargs):
    """Fill between `ylow` and `yhigh`

    Parameters
    ----------
    x : array-like, optional
    ylow : TODO, optional
    yhigh : TODO, optional

    Returns
    -------
    Chart

    """
    plot = line(x=x, y=ylow, label=label, color=color, width=width, dash=dash,
                opacity=opacity, mode=mode, fill=None, **kargs)
    plot += line(x=x, y=yhigh, label=label, color=color, width=width, dash=dash,
                 opacity=opacity, mode=mode, fill='tonexty', **kargs)
    return plot


def rug(x, label=None, opacity=None):
    """Rug chart.

    Parameters
    ----------
    x : array-like, optional
    label : TODO, optional
    opacity : TODO, optional

    Returns
    -------
    Chart

    """
    x = _try_pydatetime(x)
    x = np.atleast_1d(x)
    data = [go.Scatter(x=x, y=np.ones_like(x), name=label,
                       opacity=opacity,
                       mode='markers',
                       marker=dict(symbol='line-ns-open'))]
    layout = dict(barmode='overlay',
                  hovermode='closest',
                  legend=dict(traceorder='reversed'),
                  xaxis1=dict(zeroline=False),
                  yaxis1=dict(domain=[0.85, 1],
                              showline=False,
                              showgrid=False,
                              zeroline=False,
                              anchor='free',
                              position=0.0,
                              showticklabels=False))
    return Chart(data=data, layout=layout)


def surface(x, y, z):
    """Surface plot.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    z : array-like, optional

    Returns
    -------
    Chart

    """
    data = [go.Surface(x=x, y=y, z=z)]
    return Chart(data=data)


def hist(x, mode='overlay', label=None, opacity=None, horz=False, histnorm=None):
    """Histogram.

    Parameters
    ----------
    x : array-like
    mode : str, optional
    label : TODO, optional
    opacity : float, optional
    horz : bool, optional
    histnorm : None, "percent", "probability", "density", "probability density", optional
        Specifies the type of normalization used for this histogram trace.
        If ``None``, the span of each bar corresponds to the number of occurrences
        (i.e. the number of data points lying inside the bins). If "percent",
        the span of each bar corresponds to the percentage of occurrences with
        respect to the total number of sample points (here, the sum of all bin
        area equals 100%). If "density", the span of each bar corresponds to the
        number of occurrences in a bin divided by the size of the bin interval
        (here, the sum of all bin area equals the total number of sample
        points). If "probability density", the span of each bar corresponds to
        the probability that an event will fall into the corresponding bin
        (here, the sum of all bin area equals 1).

    Returns
    -------
    Chart

    """
    x = np.atleast_1d(x)
    if horz:
        kargs = dict(y=x)
    else:
        kargs = dict(x=x)
    layout = dict(barmode=mode)
    data = [go.Histogram(opacity=opacity, name=label, histnorm=histnorm, **kargs)]
    return Chart(data=data, layout=layout)


def hist2d(x, y, label=None, opacity=None):
    """2D Histogram.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    label : TODO, optional
    opacity : float, optional

    Returns
    -------
    Chart

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    data = [go.Histogram2d(x=x, y=y, opacity=opacity, name=label)]
    return Chart(data=data)


class PandasPlotting(object):
    """
    These plotting tools can be accessed through dataframe instance
    accessor `.plotly`.

    Examples
    --------
    Here's an example of how to do that.

    >>> df = pd.DataFrame([[1, 2], [1, 4]])
    >>> chart = df.plotly.line()
    >>> chart.show()

    """

    def __init__(self, data):
        self._data = data
        if isinstance(data, pd.DataFrame):
            self._label = data.columns
        elif isinstance(data, pd.Series):
            self._label = data.name

    def line(self, label=None, color=None, width=None, dash=None,
             opacity=None, mode='lines+markers', fill=None, **kargs):
        if label is None:
            label = self._label
        return line(x=self._data.index, y=self._data.values, label=label,
                    color=color, width=width, dash=dash, opacity=opacity, mode=mode,
                    fill=fill, **kargs)

    def scatter(self, label=None, color=None, width=None, dash=None,
                opacity=None, mode='markers', **kargs):
        """Creates a bar chart.

        Parameters
        ----------
        label : list of strings, optional
            list of labels to override column names


        Returns
        -------
        Chart

        """
        if label is None:
            label = self._label
        return scatter(x=self._data.index, y=self._data.values, label=label,
                       color=color, width=width, dash=dash, opacity=opacity, mode=mode, **kargs)

    def bar(self, label=None, mode='group', opacity=None, **kargs):
        """Creates a bar chart.

        Parameters
        ----------
        label : list of strings, optional
            list of labels to override column names
        mode : str, optional
            'group' or 'stack'

        Returns
        -------
        Chart

        """
        if label is None:
            label = self._label
        return bar(x=self._data.index, y=self._data.values, label=label,
                   mode=mode, opacity=opacity, **kargs)

    def stack(self, mode='lines', label=None, **kargs):
        """Creates a stacked area plot.

        Parameters
        ----------
        mode : string, optional
        label : list of strings, optional
            list of labels to override column names


        Returns
        -------
        Chart

        """
        if label is None:
            label = self._label

        cum = self._data.cumsum(axis=1)
        chart = Chart()
        for lab, (_, ser), (_, orig) in zip(label, cum.iteritems(), self._data.iteritems()):
            chart += line(x=ser.index, y=ser.values, label=lab,
                          fill='tonexty', mode=mode, text=orig.values, **kargs)
        return chart

    def sparklines(self, label=None, mode='lines', percent=90, epsilon=1e-3):
        """TODO: Docstring for sparklines.

        Parameters
        ----------
        label : array-like, optional
        mode : str, optional
        percent : number, optional

        Returns
        -------
        Chart

        """
        if label is None:
            label = self._label

        div = self._data.max(axis=0) - self._data.min(axis=0) + epsilon
        center = div / 2. + self._data.min(axis=0)
        normed = (self._data - center) / div
        normed *= (percent / 100.)
        offset = np.arange(1, self._data.shape[1] + 1)
        normed += offset

        chart = line(x=self._data.index, y=normed, mode=mode, label=label)
        chart.ytickvals(offset)
        chart.yticktext(self._data.columns.values)
        chart.legend(False)
        return chart


# pylint: disable=too-few-public-methods
class _AccessorProperty(object):
    """Descriptor for implementing accessor properties.
    Borrowed from pandas.
    """
    def __init__(self, accessor_cls, construct_accessor):
        self.accessor_cls = accessor_cls
        self.construct_accessor = construct_accessor
        self.__doc__ = accessor_cls.__doc__

    def __get__(self, instance, owner=None):
        if instance is None:
            return self.accessor_cls
        return self.construct_accessor(instance)

    def __set__(self, instance, value):
        raise AttributeError("can't set attribute")

    def __delete__(self, instance):
        raise AttributeError("can't delete attribute")


pd.DataFrame.plotly = _AccessorProperty(PandasPlotting, PandasPlotting)
pd.Series.plotly = _AccessorProperty(PandasPlotting, PandasPlotting)
