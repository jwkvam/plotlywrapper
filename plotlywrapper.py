"""plotly wrapper to make easy plots easy to make"""

from tempfile import NamedTemporaryFile

# pylint: disable=redefined-builtin
from builtins import zip

import plotly.offline as py
import plotly.graph_objs as go

import numpy as np
import pandas as pd

from IPython import get_ipython
from ipykernel import zmqshell


__version__ = '0.0.19'


def _labels(base='trace'):
    i = 0
    while True:
        yield base + ' ' + str(i)
        i += 1


def _detect_notebook():
    """
    this isn't 100% correct but seems good enough
    """
    kernel = get_ipython()
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
        x = x.to_pydatetime()
    except AttributeError:
        pass
    return x


class _Chart(object):
    def __init__(self, data=None, layout=None, repr_plot=True):
        self.repr_plot = repr_plot
        self.data = data
        if data is None:
            self.data = []
        self.layout = layout
        if layout is None:
            self.layout = {}
        self.figure_ = None

    def __add__(self, other):
        self.data += other.data
        self.layout = _merge_layout(self.layout, other.layout)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def group(self):
        self.layout['barmode'] = 'group'
        return self

    def stack(self):
        self.layout['barmode'] = 'stack'
        return self

    def xlabel(self, label):
        self.layout['xaxis'] = {'title': label}
        return self

    def ylabel(self, label):
        self.layout['yaxis'] = {'title': label}
        return self

    def xlim(self, low, high):
        self.layout['xaxis'] = {'range': [low, high]}
        return self

    def ylim(self, low, high):
        self.layout['yaxis'] = {'range': [low, high]}
        return self

    def title(self, string):
        self.layout['title'] = string
        return self

    def __repr__(self):
        if self.repr_plot:
            self.show(filename=None, auto_open=False)
        return super(_Chart, self).__repr__()

    def show(self, filename=None, show_link=True, auto_open=True):
        is_notebook = _detect_notebook()
        kargs = {}
        if is_notebook:
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

    def save(self, filename=None, show_link=True, auto_open=False):
        if filename is None:
            filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
        self.figure_ = go.Figure(data=self.data, layout=go.Layout(**self.layout))
        py.plot(self.figure_, show_link=show_link, filename=filename, auto_open=auto_open)
        return filename

    def to_json(self):
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


def vertical(x, ymin=0, ymax=1, color=None, width=None, dash=None, opacity=None):
    """Draws a vertical line"""
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
    return _Chart(layout=layout)


def horizontal(y, xmin=0, xmax=1, color=None, width=None, dash=None, opacity=None):
    """Draws a horizontal line"""
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
    return _Chart(layout=layout)


def line(x=None, y=None, label=None, color=None, width=None, dash=None, opacity=None,
         mode='lines+markers', fill=None):
    assert x is not None or y is not None, "x or y must be something"
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
        data = [go.Scatter(x=x, y=yy, name=ll, line=lineattr, mode=mode,
                           fill=fill, opacity=opacity)
                for ll, yy in zip(label, y.T)]
    else:
        data = [go.Scatter(x=x, y=y, name=label, line=lineattr, mode=mode,
                           fill=fill, opacity=opacity)]
    return _Chart(data=data)


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
    return _Chart(data=data)


def scatter(x=None, y=None, label=None, color=None, width=None, dash=None, opacity=None,
            mode='markers'):
    return line(x=x, y=y, label=label, color=color, width=width, dash=dash,
                mode=mode, opacity=opacity)


def bar(x=None, y=None, label=None, mode='group', opacity=None):
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

    """
    assert x is not None or y is not None, "x or y must be something"
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
        data = [go.Bar(x=x, y=yy, name=ll, opacity=opacity) for ll, yy in zip(label, y.T)]
    else:
        data = [go.Bar(x=x, y=y, name=label, opacity=opacity)]
    layout = {'barmode': mode}
    return _Chart(data=data, layout=layout)


def fill_zero(x=None, y=None, label=None, color=None, width=None, dash=None, opacity=None,
              mode='lines+markers', **kargs):
    return line(x=x, y=y, label=label, color=color, width=width, dash=dash,
                opacity=opacity, mode=mode, fill='tozeroy', **kargs)


def fill_between(x=None, ylow=None, yhigh=None, label=None, color=None, width=None, dash=None,
                 opacity=None, mode='lines+markers', **kargs):
    plot = line(x=x, y=ylow, label=label, color=color, width=width, dash=dash,
                opacity=opacity, mode=mode, fill=None, **kargs)
    plot += line(x=x, y=yhigh, label=label, color=color, width=width, dash=dash,
                 opacity=opacity, mode=mode, fill='tonexty', **kargs)
    return plot


def rug(x, label=None, opacity=None):
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
    return _Chart(data=data, layout=layout)


def surface(x, y, z):
    data = [go.Surface(x=x, y=y, z=z)]
    return _Chart(data=data)


def hist(x, mode='overlay', opacity=None, horz=False):
    if horz:
        kargs = dict(y=x)
    else:
        kargs = dict(x=x)
    layout = dict(barmode=mode)
    data = [go.Histogram(opacity=opacity, **kargs)]
    return _Chart(data=data, layout=layout)


class _PandasPlotting(object):

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
        if label is None:
            label = self._label
        return scatter(x=self._data.index, y=self._data.values, label=label,
                       color=color, width=width, dash=dash, opacity=opacity, mode=mode, **kargs)

    def bar(self, label=None, mode='group', opacity=None, **kargs):
        if label is None:
            label = self._label
        return bar(x=self._data.index, y=self._data.values, label=label,
                   mode=mode, opacity=opacity, **kargs)


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


pd.DataFrame.plotly = _AccessorProperty(_PandasPlotting, _PandasPlotting)
pd.Series.plotly = _AccessorProperty(_PandasPlotting, _PandasPlotting)
