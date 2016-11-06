.. raw:: html

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

Examples
========

Here are some more examples to highlight additional functionality.

Twin Axes
---------

.. code-block:: python

    chart = pw.bar(range(20, 15, -1))
    chart += pw.line(range(5), yaxis=2)
    chart.yaxis_right(2)

.. raw:: html
   :file: fig_twinx.html
