"""plotly wrapper to make easy plots easy to make"""

__version__ = '0.0.4'

from tempfile import NamedTemporaryFile

import plotly.offline as py
import plotly.graph_objs as go

import numpy as np

from IPython import get_ipython
from ipykernel import zmqshell


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

        fig = go.Figure(data=self.data, layout=go.Layout(**self.layout))
        plot(fig, show_link=show_link, **kargs)


class Line(_Chart):
    def __init__(self, x=None, y=None, label=None, color=None, width=None, dash=None, **kargs):
        line = {}
        if color:
            line['color'] = color
        if width:
            line['width'] = width
        if dash:
            line['dash'] = dash
        if x is None:
            x = np.arange(len(y))
        else:
            x = _try_pydatetime(x)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if y.ndim == 2:
            data = [go.Scatter(x=x, y=yy, name=label, line=line) for yy in y.T]
        else:
            data = [go.Scatter(x=x, y=y, name=label, line=line)]
        super(Line, self).__init__(data=data)


class Bar(_Chart):
    def __init__(self, x=None, y=None, label=None, mode='group', **kargs):
        if x is None:
            x = np.arange(len(y))
        else:
            x = _try_pydatetime(x)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        data = [go.Bar(x=x, y=y, name=label)]
        layout = {'barmode': 'group'}
        super(Bar, self).__init__(data=data, layout=layout)
