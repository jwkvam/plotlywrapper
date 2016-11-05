#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotlywrapper as pw
import numpy as np

options = dict(output='file',
               plotlyjs=False,
               show_link=False)

data = range(1, 6)
data2 = range(2, 12, 2)

def bar():
    pw.bar(data).save('fig_bar.html', **options)

def line():
    pw.line(data).save('fig_line.html', **options)

def scatter():
    pw.scatter(data).save('fig_scatter.html', **options)

def hist():
    pw.hist(np.sin(np.linspace(0, 2*np.pi, 500))).save('fig_hist.html', **options)

def fill_zero():
    pw.fill_zero(data).save('fig_zero.html', **options)

def fill_between():
    pw.fill_between(range(5), data, data2).save('fig_between.html', **options)

if __name__ == "__main__":
    line()
    scatter()
    bar()
    hist()
    fill_zero()
    fill_between()

