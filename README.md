# Plotlywrapper: pythonic plotly

[![Build Status](https://travis-ci.org/jwkvam/plotlywrapper.svg?branch=master)](https://travis-ci.org/jwkvam/plotlywrapper)
[![rtd.io](http://readthedocs.org/projects/plotlywrapper/badge/?version=latest)](http://plotlywrapper.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/plotlywrapper.svg)](https://badge.fury.io/py/plotlywrapper)
[![PyPI](https://img.shields.io/pypi/pyversions/plotlywrapper.svg)](https://pypi.python.org/pypi/plotlywrapper/)
[![codecov.io](https://codecov.io/github/jwkvam/plotlywrapper/coverage.svg?branch=master)](https://codecov.io/github/jwkvam/plotlywrapper?branch=master)
[![Code Health](https://landscape.io/github/jwkvam/plotlywrapper/master/landscape.svg?style=flat)](https://landscape.io/github/jwkvam/plotlywrapper/master)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/jwkvam/plotlywrapper/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/jwkvam/plotlywrapper/?branch=master)

Plotlywrapper wraps [plotly](https://plot.ly/python/) to make easy plots easy to make.
Check out the [docs](http://plotlywrapper.readthedocs.io/en/latest/)!

<p align="center">
<img width="826" alt="2D Brownian Bridge" src="https://cloud.githubusercontent.com/assets/86304/17239866/2c4c30b2-551c-11e6-9bb8-7ed467ebdacb.png">
<br>
2D <a href='https://en.wikipedia.org/wiki/Brownian_bridge'>Brownian Bridge</a>
</p>

## Motivation

Compare the following code snippets which generate the same plot.

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
plot = pw.line(x=[1, 2, 3, 4], y=[4, 1, 3, 7])
plot.title('hello world')
plot.show()
```

## Installation

To install the latest release

```
pip install plotlywrapper
```

## Demo

Try out the interactive demo here,

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/jwkvam/plotlywrapper)

or view the [notebook](http://nbviewer.jupyter.org/github/jwkvam/plotlywrapper/blob/master/index.ipynb) statically.

## Testing

To test run

```
make test
```
