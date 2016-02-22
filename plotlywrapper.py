from tempfile import NamedTemporaryFile

import plotly.offline as py
import plotly.graph_objs as go

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


class _Plot(object):
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

    def show(self, filename=None):
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

        fig = go.Figure(data=self.data, layout=go.Layout(**self.layout))
        plot(fig, **kargs)


class Scatter(_Plot):
    def __init__(self, x, y, label=None, **kargs):
        data = [go.Scatter(x=x, y=y, name=label)]
        super(Scatter, self).__init__(data=data)


class Bar(_Plot):
    def __init__(self, x, y, label=None, mode='group', **kargs):
        data = [go.Bar(x=x, y=y, name=label)]
        layout = {'barmode': 'group'}
        super(Bar, self).__init__(data=data, layout=layout)


