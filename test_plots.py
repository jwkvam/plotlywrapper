#!/usr/bin/env python
# -*- coding: utf-8 -*-

from builtins import str

import plotlywrapper as pw
import numpy as np
import pandas as pd
from numpy import random as rng

def compare_figs(d1, d2):
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    for k in d1:
        assert k in d2
        if isinstance(d1[k], dict):
            compare_figs(d1[k], d2[k])
        elif isinstance(d1[k], str):
            assert d1[k] == d2[k]
        elif isinstance(d1[k], np.ndarray):
            assert (d1[k] == d2[k]).all()
        elif hasattr(d1[k], '__iter__'):
            for v1, v2 in zip(d1[k], d2[k]):
                compare_figs(v1, v2)
        else:
            assert d1[k] == d2[k]


def test_one():
    x = np.arange(3)

    bars = pw.bar(x=x, y=[20, 14, 23], label='new york')
    bars2 = pw.bar(x=x, y=[12, 18, 29]) #, label='la')
    line = pw.line(x=x, y=[3, 8, 9], label='hello', color='red', dash='dashdot', width=5)
    plot = bars + bars2 + line
    # print(bars.data)
    plot.xlabel('x axis')
    plot.ylabel('y label')
    plot.stack()
    fig = plot.show(auto_open=False)

    expect = {'layout': {'barmode': 'stack', 'xaxis': {'title': 'x axis'},
                         'yaxis': {'title': 'y label'}},
              'data': [{'y': np.array([20, 14, 23]),
                        'x': np.array([0, 1, 2]),
                        'type': 'bar', 'name': 'new york'},
                       {'y': np.array([12, 18, 29]),
                        'x': np.array([0, 1, 2]), 'type': 'bar', 'name': None},
                       {'y': np.array([3, 8, 9]),
                        'x': np.array([0, 1, 2]), 'line': {'color': 'red', 'width': 5, 'dash': 'dashdot'},
                        'type': 'scatter', 'name': 'hello'}]}

    compare_figs(fig, expect)


def test_two():
    expect = {'layout': {'xaxis': {'title': 'x axis'}, 'yaxis': {'title': 'y label'}}, 'data': [{'y': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 'line': {'color': 'red', 'width': 5, 'dash': 'dashdot'}, 'type': 'scatter', 'name': 'hello'}]}

    x = np.arange(10)

    line = pw.line(y=x, label='hello', color='red', dash='dashdot', width=5)
    line.xlabel('x axis')
    line.ylabel('y label')
    fig = line.show(auto_open=False)

    line = pw.line(x, label='hello', color='red', dash='dashdot', width=5)
    line.xlabel('x axis')
    line.ylabel('y label')
    fig = line.show(auto_open=False)

    compare_figs(fig, expect)

def test_dataframe_lines():
    columns = list('abc')
    x = np.arange(10)
    y = rng.randn(10, 3)
    df = pd.DataFrame(y, x, columns)

    p1 = pw.lineframe(df)
    fig1 = p1.show(auto_open=False)

    p2 = pw.line(x, y, columns)
    fig2 = p2.show(auto_open=False)

    compare_figs(fig1, fig2)

def test_dataframe_bar():
    columns = list('abc')
    x = np.arange(10)
    y = rng.randn(10, 3)
    df = pd.DataFrame(y, x, columns)

    p1 = pw.barframe(df)
    fig1 = p1.show(auto_open=False)

    p2 = pw.bar(x, y, columns)
    fig2 = p2.show(auto_open=False)

    compare_figs(fig1, fig2)
