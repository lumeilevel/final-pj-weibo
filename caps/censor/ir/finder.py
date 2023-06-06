#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 20:50
# @File     : finder.py
# @Project  : final-pj-weibo
import csv
import os
import sys
import pynlpir

csv.field_size_limit(sys.maxsize)
pynlpir.open()

for fn in os.listdir(root := "../../data/raw_tbd/weibo-data/"):
    n = pynlpir.nlpir.ImportUserDict(os.path.join(root, fn))
    print("num imported: " + str(n))

files = ["week" + str(i) + ".csv" for i in range(1, 53)]

total_count = 0
of = open("../../data/raw_tbd/all_censored.csv", 'wb')
writer = csv.writer(of, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
errors = 0
unbounderrors = 0

for f in files:
    infile = "./../" + f
    with open(infile, 'rb') as csvfile:
        count = 0
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            if row[10] != "":
                mid = row[0]
                message = row[6]
                censored = 1
                try:
                    segmented = pynlpir.segment(message)
                except UnicodeDecodeError:
                    errors += 1
                    continue
                except UnboundLocalError:
                    unbounderrors += 1
                    print("what??")
                    continue
                except:
                    print("core dump...?")
                    continue

                mString = ""
                for segment in segmented:
                    mString += segment[0] + " "

                writer.writerow([mid, censored, mString.encode("utf-8")])

                # progress

                print(str(count) + "\r")
                sys.stdout.flush()

                count += 1

        print(f + ": " + str(count))
        total_count += count

print("total: " + str(total_count))
