#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/5 19:58
# @File     : main.py
# @Project  : final-pj-weibo
import argparse
import os

import jieba
import numpy as np
import pandas as pd
import yaml

import utils


def sentimentAnalysis(config, dfs, csvs):
    dfs = utils.projection(dfs, config['projection'][args.ass - 1])
    utils.transContent(dfs)
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
    dfs = utils.projection(dfs, config['projection'][args.ass - 1])
    utils.clean_df(dfs[2])
    utils.pipeline(dfs[2].content, config['ncls'])
    utils.pca_plot(dfs[2].content, utils.tfidf)
    utils.lda_pipe(dfs[2].content, config['comp'][0], config['topw'][0])
    utils.lda_generator(dfs[2].content, config['comp'][1], config['topw'][1])
    utils.kmeans_plot(dfs[2].content)
    contents = [dfs[i].content for i in range(len(csvs))]
    stopwords = utils.getStopWords('cn.txt')
    for content in contents:
        for idx in range(len(content)):
            content[idx] = utils.remove_stopwords(content[idx], stopwords)
    # result = [jieba.tokenize(content) for content in contents]
    lst = [[[res[0] for res in list(jieba.tokenize(content[i]))] for i in range(len(content))] for content in contents]
    for ls in lst:
        utils.mcdNote(*utils.topic_model(ls, config['cls']))


def wordCloud(config, dfs, csvs):
    dfs = utils.projection(dfs, config['projection'][args.ass - 1])
    len_df = {period[:-4]: len(df) for period, df in zip(csvs, dfs)}
    stopwords = utils.getStopWords('cn.txt')
    period, values = list(len_df.keys()), list(len_df.values())
    utils.periodValue(period, values)
    for df in dfs:
        post = utils.format_content(df.content.to_string(index=False))
        utils.getWordCloud(post, stopwords, max_words=config['max_words'], max_font_size=config['max_font_size'])


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
