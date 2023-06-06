#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/5 23:43
# @File     : eda.py
# @Project  : final-pj-weibo
import jieba
from gensim import corpora, models
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


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


def lda_diction(model, feature_names, n_top_words):
    lda_dict = dict()
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        features, rank = [], []
        features.append(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words:-1]]))
        rank.append(topic[:-n_top_words - 1:-1])
        d = dict(zip(features, rank))
        lda_dict[topic_idx] = {k: v for k, v in d.items()}
    return lda_dict


def lda_pipe(data, n_components, n_top_words, n_features=100):
    def print_top_words(model, feature_names, n_top_words):
        terms = []
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words:-1]]))
            terms.append(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words:-1]]))
        return terms

    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=.01, max_features=n_features)

    tf = tf_vectorizer.fit_transform(data)

    lda = LatentDirichletAllocation(n_components=n_components, max_iter=10,
                                    learning_method='online', learning_offset=50.,
                                    random_state=1)

    lda.fit(tf)

    print("\nTopics in LDA model:\n\n")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)


def lda_generator(data_samples, n_components, n_top_words):
    n_features = 100
    data_samples = data_samples
    n_components = n_components
    n_top_words = n_top_words

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=.01, ngram_range=(1, 2), max_features=n_features)

    tf_matrix = tf_vectorizer.fit_transform(data_samples)

    lda = LatentDirichletAllocation(n_components=n_components, max_iter=10,
                                    learning_method='online', learning_offset=50.,
                                    random_state=1)
    lda.fit(tf_matrix)

    feature_names = tf_vectorizer.get_feature_names()

    terms_zh = []

    for topic_idx, topic in enumerate(lda.components_):
        lda_topics = (" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words:-1]]))
        print(lda_topics)
        terms_zh.append(lda_topics)

    return terms_zh


def lst(text):
    return [[res[0] for res in list(jieba.tokenize(text[i]))] for i in range(len(text))]


def topic_model(tokens, num_topics):
    dictionary = corpora.Dictionary([i for i in tokens])
    corpus = [dictionary.doc2bow(text) for text in tokens]
    model = models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    return model, corpus, dictionary
