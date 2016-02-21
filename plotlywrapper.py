import plotly.offline as py
from plotly.graph_objs import Scatter, Layout

def scatter(x, y, title=None):
    pydata = {"data": [Scatter(x=x, y=y)]}
    if title:
        pydata['layout'] = Layout(title=title)
    py.plot(pydata)
