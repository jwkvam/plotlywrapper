from tempfile import NamedTemporaryFile

import plotly.offline as py
import plotly.graph_objs as go

from IPython import get_ipython
from ipykernel import zmqshell

DEFAULT_NAME = 'plotly_figure'

def detect_notebook():
    """
    this isn't 100% correct but seems good enough
    """
    kernel = get_ipython()
    return isinstance(kernel, zmqshell.ZMQInteractiveShell)


def _axis_label(axis, label):
    return Plot(layout={axis: {'title': label}})


def xlabel(label):
    return _axis_label('xaxis', label)

def ylabel(label):
    return _axis_label('yaxis', label)


class Plot(object):
    def __init__(self, data=None, layout=None, **kargs):
        self.data = data
        if data is None
            self.data = []
        self.layout = layout
        if layout is None
            self.layout = {}

    def __add__(self, other):
        pass

    def __radd__(self, other):
        return self.__add__(other)

    def show(self, filename=None):
        is_notebook = detect_notebook()
        kargs = {}
        if is_notebook:
            py.init_notebook_mode()
            plot = py.iplot
        else:
            plot = py.plot
            if filename is None:
                filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
            kargs['filename'] = filename

        fig = go.Figure(data=self.data, layout=go.layout(**self.layout))
        plot(fig, **kargs)


class Scatter(Plot):
    def __init__(self, x, y, label=None, **kargs):
        data = [go.Scatter(x=x, y=y, name=label)]
        super(Scatter, self).__init__(data=data)

class Bar(Plot):
    def __init__(self, x, y, label=None, mode='group', **kargs):
        data = [go.Bar(x=x, y=y, name=label)]
        layout = {barmode: 'group'}
        super(Bar, self).__init__(data=data, layout=layout)

    def group(self):
        self.layout['barmode'] = 'group'

    def stack(self):
        self.layout['barmode'] = 'stack'


