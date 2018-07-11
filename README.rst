Plotlywrapper: pythonic plotly
==============================

|Build Status| |rtd.io| |PyPI version| |PyPI| |codecov.io| |Code Health|
|Scrutinizer Code Quality|

**News** Tests are failing due to release of plotly v3.
Need to reevaluate need of library, in the meantime it still works with plotly v2.

Plotlywrapper wraps `plotly <https://plot.ly/python/>`__ to make easy
plots easy to make. Check out the
`docs <http://plotlywrapper.readthedocs.io/en/latest/>`__!

.. figure:: https://cloud.githubusercontent.com/assets/86304/17239866/2c4c30b2-551c-11e6-9bb8-7ed467ebdacb.png
   :width: 826px

   2D `Brownian Bridge <https://en.wikipedia.org/wiki/Brownian_bridge/>`__

Motivation
----------

Compare the following code snippets which generate the same plot.

Plotly
~~~~~~

Taken from https://plot.ly/python/getting-started/

.. code:: python

    import plotly
    from plotly.graph_objs import Scatter, Layout
    plotly.offline.plot({
    "data": [
        Scatter(x=[1, 2, 3, 4], y=[4, 1, 3, 7])
    ],
    "layout": Layout(
        title="hello world"
    )
    })

Plotlywrapper
~~~~~~~~~~~~~

.. code:: python

    import plotlywrapper as pw
    plot = pw.line(x=[1, 2, 3, 4], y=[4, 1, 3, 7])
    plot.title('hello world')
    plot.show()

Install
-------

To install the latest release::

    pip install plotlywrapper

Demo
----

Try out the interactive demo here,

|Binder|

or view the
`notebook <http://nbviewer.jupyter.org/github/jwkvam/plotlywrapper/blob/master/index.ipynb>`__
statically.

JupyterLab
----------

Plotly doesn’t render in JupyterLab by default. You need to install the
JupyterLab Plotly extension::

    jupyter labextension install @jupyterlab/plotly-extension

Developed in this
`repo <https://github.com/jupyterlab/jupyter-renderers>`__.

Testing
-------

To test run::

    make test

.. |Build Status| image:: https://travis-ci.org/jwkvam/plotlywrapper.svg?branch=master
   :target: https://travis-ci.org/jwkvam/plotlywrapper
.. |rtd.io| image:: http://readthedocs.org/projects/plotlywrapper/badge/?version=latest
   :target: http://plotlywrapper.readthedocs.io/en/latest/
.. |PyPI version| image:: https://badge.fury.io/py/plotlywrapper.svg
   :target: https://badge.fury.io/py/plotlywrapper
.. |PyPI| image:: https://img.shields.io/pypi/pyversions/plotlywrapper.svg
   :target: https://pypi.python.org/pypi/plotlywrapper/
.. |codecov.io| image:: https://codecov.io/github/jwkvam/plotlywrapper/coverage.svg?branch=master
   :target: https://codecov.io/github/jwkvam/plotlywrapper?branch=master
.. |Code Health| image:: https://landscape.io/github/jwkvam/plotlywrapper/master/landscape.svg?style=flat
   :target: https://landscape.io/github/jwkvam/plotlywrapper/master
.. |Scrutinizer Code Quality| image:: https://scrutinizer-ci.com/g/jwkvam/plotlywrapper/badges/quality-score.png?b=master
   :target: https://scrutinizer-ci.com/g/jwkvam/plotlywrapper/?branch=master
.. |Binder| image:: http://mybinder.org/badge.svg
   :target: http://mybinder.org/repo/jwkvam/plotlywrapper
