#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=redefined-builtin
from builtins import str

import numpy as np
from numpy import random as rng
import pandas as pd
import pytest

import plotlywrapper as pw


def compare_figs(d1, d2):
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    for k in set(d1.keys()).union(set(d2.keys())):
        assert k in d1
        assert k in d2
        if isinstance(d1[k], dict):
            compare_figs(d1[k], d2[k])
        elif isinstance(d1[k], str):
            assert d1[k] == d2[k]
        elif isinstance(d1[k], np.ndarray):
            assert (d1[k] == d2[k]).all()
        elif hasattr(d1[k], '__iter__'):
            for v1, v2 in zip(d1[k], d2[k]):
                if isinstance(v1, dict):
                    compare_figs(v1, v2)
                else:
                    assert v1 == v2
        else:
            assert d1[k] == d2[k]


def test_no_args():
    with pytest.raises(AssertionError):
        pw.line()
    with pytest.raises(AssertionError):
        pw.bar()


def test_return_none():
    x = np.arange(3)
    ret = pw.line(x).show(auto_open=False)
    assert ret is None


def test_tojson():
    js = pw.line(range(3)).to_json()
    # print(js)
    # assert set(js.keys()) == set(['data', 'layout'])
    expected = {'layout': {},
                'data': [{'opacity': None,
                          'name': None,
                          'mode': 'lines+markers',
                          'marker': dict(size=6),
                          'text': "",
                          'y': [0, 1, 2],
                          'x': [0, 1, 2],
                          'line': {},
                          'yaxis': 'y1',
                          'type': 'scatter',
                          'fill': None}]}
    compare_figs(js, expected)


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
    plot.show(auto_open=False)

    expect = {'layout': {'barmode': 'stack', 'xaxis': {'title': 'x axis'},
                         'yaxis1': {'title': 'y label'}},
              'data': [{'y': np.array([20, 14, 23]),
                        'x': np.array([0, 1, 2]),
                        'opacity': None,
                        'type': 'bar',
                        'yaxis': 'y1',
                        'name': 'new york'},
                       {'y': np.array([12, 18, 29]),
                        'x': np.array([0, 1, 2]), 'type': 'bar',
                        'opacity': None,
                        'yaxis': 'y1',
                        'name': None},
                       {'y': np.array([3, 8, 9]),
                        'x': np.array([0, 1, 2]),
                        'line': {'color': 'red', 'width': 5, 'dash': 'dashdot'},
                        'type': 'scatter',
                        'marker': dict(size=6),
                        'fill': None,
                        'yaxis': 'y1',
                        'text': "",
                        'mode': 'lines+markers',
                        'opacity': None,
                        'name': 'hello'}]}

    compare_figs(plot.figure_, expect)


def test_two():
    expect = {'layout': {'xaxis': {'title': 'x axis'}, 'yaxis1': {'title': 'y label'}},
              'data': [{'y': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                        'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                        'line': {'color': 'red', 'width': 5, 'dash': 'dashdot'},
                        'marker': dict(size=6),
                        'type': 'scatter',
                        'fill': None,
                        'text': "",
                        'yaxis': 'y1',
                        'mode': 'lines+markers',
                        'opacity': None,
                        'name': 'hello'}]}

    x = np.arange(10)

    line0 = pw.line(y=x, label='hello', color='red', dash='dashdot', width=5)
    line0.xlabel('x axis')
    line0.ylabel('y label')
    line0.show(auto_open=False)

    line1 = pw.line(x, label='hello', color='red', dash='dashdot', width=5)
    line1.xlabel('x axis')
    line1.ylabel('y label')
    line1.show(auto_open=False)

    compare_figs(line0.figure_, line1.figure_)
    compare_figs(line0.figure_, expect)

def test_dataframe_lines():
    columns = list('abc')
    x = np.arange(10)
    y = rng.randn(10, 3)
    df = pd.DataFrame(y, x, columns)

    p1 = df.plotly.line()
    p1.show(auto_open=False)

    p2 = pw.line(x, y, columns)
    p2.show(auto_open=False)

    compare_figs(p1.figure_, p2.figure_)

def test_dataframe_bar():
    columns = list('abc')
    x = np.arange(10)
    y = rng.randn(10, 3)
    df = pd.DataFrame(y, x, columns)

    p1 = df.plotly.bar()
    p1.show(auto_open=False)

    p2 = pw.bar(x, y, columns)
    p2.show(auto_open=False)

    compare_figs(p1.figure_, p2.figure_)
