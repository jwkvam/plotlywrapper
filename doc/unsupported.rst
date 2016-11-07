Unsupported Features
====================

Plotlywrapper implements a small subset of Plotly's API.
If a feature isn't implemented, you can still use Plotlywrapper and modify the data structure yourself.

Data Structure
--------------

First let's learn the data structure.
Plotly has two main data sources for their plots: a list of traces and layout options.
The list of traces is a list of simple plot types and their data such as scatter plots, histograms, and surface plots.
The `Chart` class stores this in its `data` field.
The layout is a dictionary of options that globally affect the plot.
The `Chart` class stores this in its `layout` field.

Examples
--------

Y Axis Format
~~~~~~~~~~~~~

Change the Y Axis `type <https://plot.ly/python/reference/#layout-yaxis-type>`_ to log::

    chart = line(np.exp(np.arange(10)))
    chart.layout['yaxis1']['type'] = 'log'
