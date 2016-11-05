#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotlywrapper as pw

options = dict(output='file',
               plotlyjs=False,
               show_link=False)

data = range(1, 6)

def bar():
    pw.bar(data).save('fig_bar.html', **options)

def line():
    pw.line(data).save('fig_line.html', **options)

def scatter():
    pw.scatter(data).save('fig_scatter.html', **options)

if __name__ == "__main__":
    line()
    scatter()
    bar()
