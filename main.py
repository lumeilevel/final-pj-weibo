#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/5 19:58
# @File     : main.py
# @Project  : final-pj-weibo
import argparse
import os

import numpy as np
import pandas as pd
import yaml

import utils


def sentimentAnalysis(config, dfs, csvs):
    dfs = utils.projection(dfs, config['projection'][args.ass-1])
    utils.transContent(dfs)
    print(type(dfs[0]))
    print(dfs[0].columns.values)
    mean = utils.addSentiment(dfs, config['col']['sentiment'])
    for m in mean:
        print(f"Mean sentiment of {csvs[mean.index(m)][:-4]} is {m}")
    utils.addDate(dfs, config['col']['date'])
    sentiTime = []
    # add heat and print sentiment
    for i, df in enumerate(dfs):
        df['heat'] = df['forwards'] + df['comments'] + df['likes']
        sentiMean = df.groupby(df['date']).apply(lambda x: np.average(x.sentiment, weights=x['heat']))
        sentiTime.append(sentiMean)
        print(f"{csvs[i][:-4]}: {sentiMean}")
    # Visualization
    utils.sentimentTrend(mean, config['figsize'])
    utils.sentimentTrendDaily(sentiTime, csvs, **config['vis'])


def topicModeling(config, dfs, csvs):
    pass


def wordCloud(config, dfs, csvs):
    pass


def main(config):
    ass = {1: sentimentAnalysis, 2: topicModeling, 3: wordCloud}
    csvs = os.listdir(df_dir := f"data/{config['tdir']}")
    dfs = [pd.read_csv(os.path.join(df_dir, fn)) for fn in csvs]
    ass[args.ass](config, dfs, csvs)


if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description="Let's dig deeper into the world of Weibo!")
    parser.add_argument('--seed', '-s', default=config['seed'], type=int, help="Set random seed")
    parser.add_argument('--ass', '-a', default=config['ass'], type=int, choices=range(1, 4),
                        help="1 for sentiment analysis; 2 for topic modeling; 3 for word cloud demonstration.")
    args = parser.parse_args()
    main(config)
