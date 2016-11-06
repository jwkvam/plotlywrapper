.. raw:: html

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

Examples
========

Here are some more examples to highlight additional functionality.

Bubble
------

.. code-block:: python

    scatter(range(1, 6), markersize=numpy.arange(1, 6) * 10)

.. raw:: html
   :file: fig_bubble.html

Twin Axes
---------

.. code-block:: python

    chart = bar(range(20, 15, -1))
    chart += line(range(5), yaxis=2)
    chart.yaxis_right(2)

.. raw:: html
   :file: fig_twinx.html
