#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotlywrapper as pw
import numpy as np
import pandas as pd
# from numpy import random as rng
import random
random.seed(0)

# rng.seed(0)

options = dict(output='file',
               plotlyjs=False,
               show_link=False)

datas = 'range(1, 6)'
data = eval(datas)
data2s = 'range(2, 12, 2)'
data2 = eval(data2s)

def bar():
    pw.bar(data).save('fig_bar.html', **options)

def line():
    pw.line(data).save('fig_line.html', **options)

def scatter():
    pw.scatter(data).save('fig_scatter.html', **options)

def hist():
    pw.hist(np.sin(np.linspace(0, 2*np.pi, 100))).save('fig_hist.html', **options)

def heatmap():
    pw.heatmap(np.arange(25).reshape(5, -1)).save('fig_heatmap.html', **options)

def heatmap2():
    x = np.arange(5)
    pw.heatmap(z=np.arange(25), x=np.tile(x, 5), y=x.repeat(5)).save('fig_heatmap2.html', **options)

def hist2d():
    pw.hist2d(np.sin(np.linspace(0, 2*np.pi, 100)),
              np.cos(np.linspace(0, 2*np.pi, 100))).save('fig_hist2d.html', **options)

def fill_zero():
    chart = pw.fill_zero(data).save('fig_zero.html', **options)

def fill_between():
    pw.fill_between(range(5), data, data2).save('fig_between.html', **options)

def twin_axes():
    chart = pw.bar(range(20, 15, -1))
    chart += pw.line(range(5), yaxis=2)
    chart.yaxis_right(2)
    chart.save('fig_twinx.html', **options)

def bubble():
    chart = pw.scatter(data, markersize=np.arange(1, 6) * 10)
    chart.save('fig_bubble.html', **options)

def parallel():
    df = pd.DataFrame([[1,3,2], [2,1,3]], columns=['alpha', 'beta', 'gamma'])
    chart = df.T.plotly.line()
    chart.legend(False)
    chart.save('fig_parallel.html', **options)

if __name__ == "__main__":
    line()
    scatter()
    bar()
    hist()
    hist2d()
    fill_zero()
    fill_between()
    twin_axes()
    bubble()
    heatmap()
    heatmap2()
    parallel()
