#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotlywrapper as pw
import numpy as np
from numpy import random as rng

# pw.scatter(range(5), range(5))
# pw.scatter(np.arange(5), np.arange(5))

x = rng.randn(10)

line = pw.Scatter(y=x, label='hello', color='red', dash='dashdot', width=5)
# print(bars.data)
line.xlabel('x axis')
line.ylabel('y label')
line.show()
