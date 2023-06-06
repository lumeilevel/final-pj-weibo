#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 21:00
# @File     : mapf.py
# @Project  : final-pj-weibo
import csv
from collections import defaultdict

infile = "../../data/raw_tbd/all_censored.csv"
f = open(infile, 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
userDict = defaultdict(int)

for row in reader:
    userDict[row[4]] += 1

for key, value in userDict.iteritems():
    print(str(key) + ":" + str(value))
