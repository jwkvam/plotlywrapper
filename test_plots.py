"""Tests for Plotlywrapper."""
import json
from datetime import datetime, time, date

import numpy as np
from numpy import random as rng
import pandas as pd
import pytest

import plotlywrapper as pw


def json_conversion(obj):
    """Encode additional objects to JSON."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, pd.DatetimeIndex):
        return [x.isoformat() for x in obj.to_pydatetime()]
    if isinstance(obj, pd.Index):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        try:
            return [x.isoformat() for x in obj.dt.to_pydatetime()]
        except AttributeError:
            return obj.tolist()
    if isinstance(obj, (datetime, time, date)):
        return obj.isoformat()
    raise TypeError('Not sure how to serialize {} of type {}'.format(obj, type(obj)))


def compare_figs(d1, d2):
    """Compare charts."""
    d1 = json.loads(json.dumps(d1, default=json_conversion))
    d2 = json.loads(json.dumps(d2, default=json_conversion))
    _compare_figs(d1, d2)


def _compare_figs(d1, d2):
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    for k in set(d1.keys()).union(set(d2.keys())):
        if k == 'uid':
            continue
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
    """Test no args raises an error."""
    with pytest.raises(AssertionError):
        pw.line()
    with pytest.raises(AssertionError):
        pw.bar()


def test_dict():
    """Test dict accessor works."""
    js = pw.line(range(3)).dict
    expected = {
        'layout': {},
        'data': [
            {
                'mode': 'lines+markers',
                'marker': dict(size=6),
                'text': "",
                'y': [0, 1, 2],
                'x': [0, 1, 2],
                'yaxis': 'y',
                'type': 'scatter',
            }
        ],
    }
    compare_figs(js, expected)


def test_one():
    """First charting test."""
    x = np.arange(3)

    bars = pw.bar(x=x, y=[20, 14, 23], label='new york')
    bars2 = pw.bar(x=x, y=[12, 18, 29])  # , label='la')
    line = pw.line(x=x, y=[3, 8, 9], label='hello', color='red', dash='dashdot', width=5)
    plot = bars + bars2 + line
    # print(bars.data)
    plot.xlabel = 'x axis'
    plot.ylabel = 'y label'
    plot.stack()
    plot.show(auto_open=False)

    expect = {
        'layout': {'barmode': 'stack', 'xaxis': {'title': 'x axis'}, 'yaxis': {'title': 'y label'}},
        'data': [
            {
                'y': np.array([20, 14, 23]),
                'x': np.array([0, 1, 2]),
                'type': 'bar',
                'yaxis': 'y',
                'name': 'new york',
            },
            {'y': np.array([12, 18, 29]), 'x': np.array([0, 1, 2]), 'type': 'bar', 'yaxis': 'y'},
            {
                'y': np.array([3, 8, 9]),
                'x': np.array([0, 1, 2]),
                'line': {'color': 'red', 'width': 5, 'dash': 'dashdot'},
                'type': 'scatter',
                'marker': dict(size=6),
                'yaxis': 'y',
                'text': "",
                'mode': 'lines+markers',
                'name': 'hello',
            },
        ],
    }

    compare_figs(plot.dict, expect)


def test_two():
    """Second charting test."""
    expect = {
        'layout': {'xaxis': {'title': 'x axis'}, 'yaxis': {'title': 'y label'}},
        'data': [
            {
                'y': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                'line': {'color': 'red', 'width': 5, 'dash': 'dashdot'},
                'marker': dict(size=6),
                'type': 'scatter',
                'text': "",
                'yaxis': 'y',
                'mode': 'lines+markers',
                'name': 'hello',
            }
        ],
    }

    x = np.arange(10)

    line0 = pw.line(y=x, label='hello', color='red', dash='dashdot', width=5)
    line0.xlabel = 'x axis'
    line0.ylabel = 'y label'
    line0.show(auto_open=False)

    line1 = pw.line(x, label='hello', color='red', dash='dashdot', width=5)
    line1.xlabel = 'x axis'
    line1.ylabel = 'y label'
    line1.show(auto_open=False)

    compare_figs(line0.dict, line1.dict)
    compare_figs(line0.dict, expect)


def test_dataframe_lines():
    """Test dataframe lines chart."""
    columns = list('abc')
    x = np.arange(10)
    y = rng.randn(10, 3)
    df = pd.DataFrame(y, x, columns)

    p1 = df.plotly.line()
    p1.show(auto_open=False)

    p2 = pw.line(x, y, columns)
    p2.show(auto_open=False)

    compare_figs(p1.dict, p2.dict)


def test_dataframe_bar():
    """Test dataframe bar chart."""
    columns = list('abc')
    x = np.arange(10)
    y = rng.randn(10, 3)
    df = pd.DataFrame(y, x, columns)

    p1 = df.plotly.bar()
    p1.show(auto_open=False)

    p2 = pw.bar(x, y, columns)
    p2.show(auto_open=False)

    compare_figs(p1.dict, p2.dict)
