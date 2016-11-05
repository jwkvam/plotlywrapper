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
