#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 19:55
# @File     : column2.py
# @Project  : final-pj-weibo
import csv

files = ["2014-11_133-keywords-which-cause-posts-to-be-hidden-invisible-on-weibo.csv",
         "2014-11_14-words-which-give-you-a-sync-error-when-posting-on-weibo.csv",
         "2014-11_66-words-you-cant-post-on-weibo.csv",
         "2014-11_784-CDT-noresults-words-auto-review-censorship-keywords.csv"]

index = 0

for fi in files:
    f = open(fi, 'rb')
    of = open(str(index) + ".txt", 'wb')
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        of.write(row[2] + "\n")

    index += 1

f = open(fi := "2014-11_2429-CDT-words-search-censorship.csv", 'rb')
of = open(str(index) + ".txt", 'wb')
reader = csv.reader(f, delimiter=",")
for row in reader:
    of.write(row[1] + "\n")

index += 1
