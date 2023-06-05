#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/5 21:47
# @File     : visualization.py
# @Project  : final-pj-weibo
import os
from datetime import datetime

import pandas as pd
from bokeh import palettes
from bokeh.io import show, output_notebook, export_png
from bokeh.plotting import figure
from matplotlib import pyplot as plt


def sentimentTrend(sentiment, figsize=(12, 9), log_dir='log/visualization/'):
    plt.figure(figsize=figsize)
    plt.bar(["Before congress", "During congress", "After congress"], sentiment, color='lightgreen', width=0.25)
    plt.plot(["Before congress", "During congress", "After congress"], sentiment, color='gray', marker='o')
    plt.ylabel("Daily Average Sentiment Score")
    plt.title("Sentiment Trend")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(log_dir, f"sentiment_trend_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"), dpi=300)
    plt.show()


def sentimentTrendDaily(sentiment, csvs, log_dir='log/visualization/', **kwargs):
    output_notebook()
    p = figure(x_axis_type='datetime', width=kwargs['width'], height=kwargs['height'], title="Sentiment Daily Trend")
    # fcbba1 | c6dbef | c7e9c0
    colors = palettes.Blues9
    days = [pd.DataFrame(s, columns=['sentiment_score']) for s in sentiment]
    for i, df in enumerate(days):
        p.vbar(x=df.index, top=df['sentiment_score'], color=colors[i],
               legend_label=f'sentiment score_{csvs[i][:-4]}', alpha=kwargs['alpha'], width=kwargs['width1'])
    p.line(x=[datetime(2022, 9, 16), datetime(2022, 10, 16)], y=[4.440892e-16, 5.376048e-01], color="red", line_width=3)
    p.line(x=[datetime(2022, 10, 17), datetime(2022, 10, 21)], y=[0.921040, 0.587548], color="green", line_width=3)
    p.line(x=[datetime(2022, 10, 22), datetime(2022, 11, 22)], y=[0.173966, 0.781571], color="blue", line_width=3)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    show(p)
    export_png(p, filename=os.path.join(log_dir,
                                        f"sentiment_trend_daily_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"))
