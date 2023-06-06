#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 21:01
# @File     : open_parse.py
# @Project  : final-pj-weibo
import csv
import sys

csv.field_size_limit(sys.maxsize)
infile = "../../data/raw_tbd/all_censored.csv"
f = open(infile, 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
count = 0
total = sum(1 for row in reader)
print(total)
f.seek(0)
cCount = 0

for row in reader:
    if len(row) > 3:
        print(row[3])
    if row[1] == "1":
        cCount += 1

print(cCount)
