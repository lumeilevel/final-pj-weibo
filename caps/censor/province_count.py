#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 19:35
# @File     : province_count.py
# @Project  : final-pj-weibo
import csv
import sys

csv.field_size_limit(sys.maxsize)
provinces = [11, 12, 13, 14, 15, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 61,
             62, 63, 64, 65, 81, 82, 100, 400]

files = ["week" + str(i) + ".csv" for i in range(1, 53)]
userDict = dict()

print("loading user data into map")
with open(userdata := "userdata.csv", 'rb') as users:
    reader = csv.reader(users, delimiter=",")
    for row in reader:
        userDict[row[0]] = [row[1], row[2], row[3]]

print(userDict['uDZPARQHJ'])

print("counting censorship by province")
province_censored_count_dict = dict()
province_not_censored_count_dict = dict()

for p in provinces:
    province_censored_count_dict[p] = 0
    province_not_censored_count_dict[p] = 0

for f in files:
    with open(f, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            try:
                d = userDict[row[2]]
                # print d[0]
                prov = int(d[0])
                if row[10] == "":
                    province_not_censored_count_dict[prov] += 1
                else:
                    province_censored_count_dict[prov] += 1
            except KeyError:
                continue
            except ValueError:
                print("ValueError")

    print(province_censored_count_dict)
    print(province_not_censored_count_dict)
    print("-" * 10)
