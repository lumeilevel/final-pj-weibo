#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/5 23:43
# @File     : eda.py
# @Project  : final-pj-weibo
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(data):
    # instantiate classifier
    tfidf_vectorizer = TfidfVectorizer(min_df=.01, max_df=.8, ngram_range=(1, 2))
    # generate matrix
    k = tfidf_vectorizer.fit_transform(data)
    return tfidf_vectorizer.fit_transform(data)  # fit the vectorizer to synopses


# run k-means on tf-idf matrix
def k_means(data, num_clusters):
    km = KMeans(n_clusters=num_clusters)
    km.fit(data)
    clusters = km.labels_.tolist()
    return km, clusters


# extract feature words
def feature_terms(data):
    tfidf_vectorizer = TfidfVectorizer(max_df=.99, min_df=.0001, ngram_range=(1, 3))
    # tfidf_vectorizer = TfidfVectorizer(max_df=.8, min_df=.01, ngram_range=(1,2))
    vectors = tfidf_vectorizer.fit(data)
    return tfidf_vectorizer.get_feature_names()


# find the terms with highest tf-idf score
def get_top_terms(km, review_terms, num_clusters):
    print("Top terms per cluster:\n")

    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(1, num_clusters):
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :10]:  # replace 11 with n words per cluster
            print(' %s' % review_terms[ind], end=',')
        print()  # add whitespace
        print()  # add whitespace

    print('\n')


# run functions sequentially
def pipeline(text, n_clusters):
    matrix = tfidf(text)
    km, clusters = k_means(matrix, n_clusters)
    features = feature_terms(text)
    get_top_terms(km, features, n_clusters)
