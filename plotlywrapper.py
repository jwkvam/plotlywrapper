"""plotly wrapper to make easy plots easy to make"""

from tempfile import NamedTemporaryFile

from builtins import zip

import plotly.offline as py
import plotly.graph_objs as go

import numpy as np

from IPython import get_ipython
from ipykernel import zmqshell


__version__ = '0.0.8'


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


def _merge_dicts(d1, d2):
    d = d2.copy()
    d.update(d1)
    return d


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
    def __init__(self, data=None, layout=None, **kargs):
        self.data = data
        if data is None:
            self.data = []
        self.layout = layout
        if layout is None:
            self.layout = {}

    def __add__(self, other):
        self.data += other.data
        self.layout = _merge_dicts(self.layout, other.layout)
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


def line(x=None, y=None, label=None, color=None, width=None, dash=None, **kargs):
    assert x is not None or y is not None, "x or y must be something"
    line = {}
    if color:
        line['color'] = color
    if width:
        line['width'] = width
    if dash:
        line['dash'] = dash
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
        data = [go.Scatter(x=x, y=yy, name=ll, line=line) for ll, yy in zip(label, y.T)]
    else:
        data = [go.Scatter(x=x, y=y, name=label, line=line)]
    return _Chart(data=data)


def lineframe(data, color=None, width=None, dash=None, **kargs):
    return line(x=data.index, y=data.values, label=data.columns,
                color=color, width=width, dash=dash, **kargs)


def bar(x=None, y=None, label=None, mode='group', **kargs):
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
        data = [go.Bar(x=x, y=yy, name=ll) for ll, yy in zip(label, y.T)]
    else:
        data = [go.Bar(x=x, y=y, name=label)]
    layout = {'barmode': mode}
    return _Chart(data=data, layout=layout)


def barframe(data, mode='group', **kargs):
    return bar(x=data.index, y=data.values, label=data.columns,
               mode=mode, **kargs)
