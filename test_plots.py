#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotlywrapper as pw
import numpy as np
from numpy import random as rng


def test_one():
    x = np.arange(3)

    bars = pw.Bar(x=x, y=[20, 14, 23], label='new york')
    bars2 = pw.Bar(x=x, y=[12, 18, 29]) #, label='la')
    line = pw.Line(x=x, y=np.random.randn(3) * 20, label='hello', color='red', dash='dashdot', width=5)
    plot = bars + bars2 + line
    # print(bars.data)
    plot.xlabel('x axis')
    plot.ylabel('y label')
    plot.stack()
    plot.show(auto_open=False)


def test_two():
    x = rng.randn(10)

    line = pw.Line(y=x, label='hello', color='red', dash='dashdot', width=5)
    # print(bars.data)
    line.xlabel('x axis')
    line.ylabel('y label')
    line.show(auto_open=False)
