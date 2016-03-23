# plotlywrapper: pythonic plotly

plotlywrapper wraps [plotly](https://plot.ly/python/) to make easy plots easy to make.

[![Build Status](https://travis-ci.org/jwkvam/plotlywrapper.svg?branch=master)](https://travis-ci.org/jwkvam/plotlywrapper)
[![PyPI version](https://badge.fury.io/py/plotlywrapper.svg)](https://badge.fury.io/py/plotlywrapper)
[![PyPI](https://img.shields.io/pypi/dm/plotlywrapper.svg)](https://badge.fury.io/py/plotlywrapper)
[![codecov.io](https://codecov.io/github/jwkvam/plotlywrapper/coverage.svg?branch=master)](https://codecov.io/github/jwkvam/plotlywrapper?branch=master)
[![Code Health](https://landscape.io/github/jwkvam/plotlywrapper/master/landscape.svg?style=flat)](https://landscape.io/github/jwkvam/plotlywrapper/master)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/jwkvam/plotlywrapper/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/jwkvam/plotlywrapper/?branch=master)

## Motivation

To understand why I made this project compare the following code snippets which generate the same plot.

### Plotly

Taken from https://plot.ly/python/getting-started/

``` python
import plotly
from plotly.graph_objs import Scatter, Layout
plotly.offline.plot({
"data": [
    Scatter(x=[1, 2, 3, 4], y=[4, 1, 3, 7])
],
"layout": Layout(
    title="hello world"
)
})
```

### Plotlywrapper
``` python
import plotlywrapper as pw
plot = pw.scatter(x=[1, 2, 3, 4], y=[4, 1, 3, 7])
plot.title('hello world')
plot.show()
```

## Installation

To install the latest release

```
pip install plotlywrapper
```

## Demo

This [notebook](http://nbviewer.jupyter.org/github/jwkvam/plotlywrapper/blob/master/demo.ipynb) shows off a portion of the API.

## Testing

To test run

```
make test
```
