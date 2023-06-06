#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/5 21:01
# @File     : preprocess.py
# @Project  : final-pj-weibo
import os
import re
import string

import jieba
import pandas as pd
from snownlp import SnowNLP


def projection(dfs, cols):
    # return [df.loc[:, cols] for df in dfs]
    return [df[cols] for df in dfs]


def transContent(dfs):
    # return list(map(lambda x: x.content.str.replace("\"", "").astype('unicode'), dfs))
    map(lambda x: x.content.str.replace("\"", "").astype('unicode'), dfs)


def addSentiment(dfs, col_name='sentiment'):
    for df in dfs:
        df[col_name] = SnowNLP(df['content'])
        for row in range(len(df)):
            df[col_name][row] = SnowNLP(df['content'][row]).sentiments
    return [df[col_name].mean() for df in dfs]


def addDate(dfs, col_name='date'):
    for df in dfs:
        # df[col_name] = df['time'].str.split(' ', expand=True)[0]
        df[col_name] = pd.to_datetime(df['time'], infer_datetime_format=True).dt.date
        df[col_name] = pd.to_datetime(df[col_name], infer_datetime_format=True)
    # return [df[col_name].unique() for df in dfs]


def cut_text(text):
    return ",".join(jieba.cut_for_search(text, HMM=True))


def remove_stopwords(text, stopwords):
    return "".join([x for x in text if x not in stopwords])


def remove_ascii(text):
    exclude = set(str(string.punctuation + string.digits + string.ascii_uppercase + string.ascii_lowercase))
    return "".join(ch for ch in text if ch not in exclude)


def remove_unicode(text):
    return re.sub(r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！\ ，。？、~@#￥%……&*（）：；《）《》“”(<>)»〔〕-]+", "", text)


def process_text(text, stopwords):
    stopwords_removed = remove_stopwords(text, stopwords)
    removed_ascii = remove_ascii(stopwords_removed)
    tokenized = cut_text(removed_ascii)
    return remove_unicode(tokenized)


def clean_df(df, threshold=2):
    return df.drop(df[df['content'].map(len) < threshold].index)


def getStopWords(filename, data_dir='data/stopwords/'):
    with open(os.path.join(data_dir, filename)) as f:
        stopword = f.readlines()
        stopwords = [line.strip('\n\r') for line in stopword]
    return stopwords


def format_content(content):
    content = content.replace(u'\xa0', u' ')
    content = re.sub(r'\[.*?\]', '', content)
    content = content.replace('\n', ' ')
    return content
