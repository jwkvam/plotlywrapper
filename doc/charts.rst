.. raw:: html

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

Charts
======

These are the basic chart types Plotlywrapper supports.


Scatter
-------

line
~~~~

.. autofunction:: plotlywrapper.line

**Examples**

.. code-block:: python

    line(range(1, 6))

.. raw:: html
   :file: fig_line.html

scatter
~~~~~~~

.. autofunction:: plotlywrapper.scatter

**Examples**

.. code-block:: python

    scatter(range(1, 6))

.. raw:: html
   :file: fig_scatter.html


Bar
---

bar
~~~

.. autofunction:: plotlywrapper.bar

**Examples**

.. code-block:: python

    bar(range(1, 6))

.. raw:: html
   :file: fig_bar.html


Histogram
---------

hist
~~~~

.. autofunction:: plotlywrapper.hist

**Examples**

.. code-block:: python

    hist(np.sin(np.linspace(0, 2*np.pi, 500)))

.. raw:: html
   :file: fig_hist.html

hist2d
~~~~~~

.. autofunction:: plotlywrapper.hist2d

**Examples**

.. code-block:: python

    hist2d(np.sin(np.linspace(0, 2*np.pi, 100)),
           np.cos(np.linspace(0, 2*np.pi, 100)))

.. raw:: html
   :file: fig_hist2d.html

Filled Area
-----------

fill_zero
~~~~~~~~~

.. autofunction:: plotlywrapper.fill_zero

**Examples**

.. code-block:: python

    bar(range(1, 6))

.. raw:: html
   :file: fig_zero.html

fill_between
~~~~~~~~~~~~

.. autofunction:: plotlywrapper.fill_between

**Examples**

.. code-block:: python

    fill_between(range(5), range(1, 6), range(2, 12, 2))

.. raw:: html
   :file: fig_between.html


Heatmap
-------

heatmap
~~~~~~~

.. autofunction:: plotlywrapper.heatmap

Note: If ``z`` is 1D, then you must specify ``x`` and ``y`` as well.
For example, taking a Pandas series with a time index, setting ``x = series.index.date`` and ``y = series.index.time``,
then z needs to be 1D (e.g. ``z = series.values``).
This would plot an empty chart if using a Pandas DataFrame,
as ``df.values`` will result in an array of [[z1],[z2],[z3]...], which is not 1D.

**Examples**

.. code-block:: python

    heatmap(np.arange(25).reshape(5, -1))

.. raw:: html
   :file: fig_heatmap.html

.. code-block:: python

    x = np.arange(5)
    heatmap(z=np.arange(25), x=np.tile(x, 5), y=x.repeat(5))

.. raw:: html
   :file: fig_heatmap2.html

3D
--

.. autofunction:: plotlywrapper.line3d
.. autofunction:: plotlywrapper.scatter3d
.. autofunction:: plotlywrapper.surface

Shapes
------

.. autofunction:: plotlywrapper.vertical
.. autofunction:: plotlywrapper.horizontal

Rug
---

.. autofunction:: plotlywrapper.rug
