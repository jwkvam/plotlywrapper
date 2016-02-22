#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotlywrapper as pw
import numpy as np

# pw.scatter(range(5), range(5))
# pw.scatter(np.arange(5), np.arange(5))

x = np.arange(3)

bars = pw.Bar(x=x, y=[20, 14, 23], label='new york')
bars2 = pw.Bar(x=x, y=[12, 18, 29], label='la')
line = pw.Scatter(x=x, y=np.random.randn(3) * 20, label='hello', color='red', dash='dashdot', width=5)
plot = bars + bars2 + line
# print(bars.data)
plot.xlabel('x axis')
plot.ylabel('y label')
plot.stack()
plot.show()
