#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/5 21:47
# @File     : visualization.py
# @Project  : final-pj-weibo
import os
from datetime import datetime

import jieba
import numpy as np
import pandas as pd
import pyLDAvis
from bokeh import palettes
from bokeh.io import show, output_notebook, export_png, save
from bokeh.plotting import figure
from matplotlib import pyplot as plt
from pyLDAvis import gensim_models
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def sentimentTrend(sentiment, figsize=(12, 9), log_dir='log/visualization/'):
    plt.figure(figsize=figsize)
    plt.bar(["Before congress", "During congress", "After congress"], sentiment, color='orange', width=0.25)
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
    colors = (palettes.Reds3[0], palettes.Greens3[0], palettes.Blues3[0])
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
    save(p,
         filename=os.path.join(log_dir, f"sentiment_trend_daily_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.html"),
         title="Sentiment Daily Trend")
    # export_png(p, filename=os.path.join(log_dir,
    #                                     f"sentiment_trend_daily_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"))


def pca_plot(text, tfidf, fig_dir='log/visualization/'):
    X = tfidf(text).todense()
    # color = ['r', 'b', 'g']
    pca = PCA(n_components=3).fit(X)
    data2D = pca.transform(X)
    plt.scatter(data2D[:, 0], data2D[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title("K-means Principal Components")
    plt.savefig(os.path.join(fig_dir + f"pca_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"), dpi=300)
    plt.show()


def kmeans_plot(data, fig_dir='log/visualization/'):
    tfidf = TfidfVectorizer(min_df=.01, ngram_range=(1, 2))
    X = tfidf.fit_transform(data).todense()

    # cluster data into K=1..10 clusters
    K = range(1, 10)

    # scipy kmeans module for each value of k:
    KM = [kmeans(X, k) for k in K]
    # list comprehension to cluster centroids
    centroids = [cent for (cent, var) in KM]

    # alternative: scipy.spatial.distance.cdist
    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D, axis=1) for D in D_k]
    dist = [np.min(D, axis=1) for D in D_k]
    avgWithinSS = [sum(d) / X.shape[0] for d in dist]

    # plot
    kIdx = 4

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='x', markersize=12,
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.savefig(os.path.join(fig_dir + f"elbow_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"), dpi=300)

    # scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #     ax.scatter(X[:,2],X[:,1], s=30, c=cIdx[k])
    clr = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(K[kIdx]):
        ind = (cIdx[kIdx] == i)
        ax.scatter([X[ind, 1]], [X[ind, 2]], s=30, c=clr[i], label='Cluster %d' % i)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Censored Tweet Clusters with K=%d' % K[kIdx])
    plt.legend()
    plt.savefig(os.path.join(fig_dir + f"kmeans_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"), dpi=300)
    plt.show()


def mcdNote(model_hl, corpus_hl, dict_hl, log_dir='log/visualization/'):
    # pyLDAvis.enable_notebook()    # for jupyter notebook
    p = gensim_models.prepare(model_hl, corpus_hl, dict_hl)
    pyLDAvis.save_html(p, os.path.join(log_dir, f"mcd_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.html"))


def periodValue(period, values, log_dir='log/visualization/'):
    plt.bar(period, values, color='maroon', width=0.4)

    # Add annotation to bars
    for i, v in enumerate(values):
        plt.text(i, v + 50, str(v))

    plt.xlabel("Period")
    plt.ylabel("Number of posts")
    plt.title("Number of Weibo posts across different periods")
    plt.savefig(os.path.join(log_dir, f"periodValue_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"), dpi=300)
    plt.show()


def getWordCloud(content, stopwords, log_dir='log/visualization/',
                 font_dir='config/', max_words=5000, max_font_size=50):
    words = jieba.lcut(content)
    new_txt = ''.join(words)
    word_cloud = WordCloud(stopwords=stopwords, font_path=os.path.join(font_dir, 'SourceHanSansCN-Normal.otf'),
                           max_words=max_words, background_color='white', max_font_size=max_font_size,
                           scale=32).generate(new_txt)
    word_cloud.to_file(os.path.join(log_dir, f"wordcloud_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.jpg"))
