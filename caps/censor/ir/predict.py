#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 21:11
# @File     : predict.py
# @Project  : final-pj-weibo
import csv
import datetime
import random
from collections import defaultdict

import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR

provinces = [11, 82, 54]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

mi_infile = "../../data/raw_tbd/mi_results.csv"
mi_read = open(mi_infile, 'rb')
reader = csv.reader(mi_read, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
rowCount = 0
feature_words = []

print()
"loading sentiment sets"
pos = open('../../data/raw_tbd/pos.txt', 'rb')
neg = open('../../data/raw_tbd/neg.txt', 'rb')
pos_set = set([])
neg_set = set([])

for line in pos:
    pos_set.add(line.rstrip())

for line in neg:
    neg_set.add(line.rstrip())
print("appending feature words")

for row in reader:
    if row[0] == "||":
        print(rowCount)

    if rowCount < 0:
        feature_words.append(row[0])
    else:
        break
    rowCount += 1

train_matrix = []
test_matrix = []

print("adding censored data")
infile = "../../data/raw_tbd/all_censored.csv"
f = open(infile, 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
censored_train_cutoff = 73567 - (73567 / float(7))  # /float(10)
userDict = defaultdict(int)
rowCount = 0
addedCount = 0
greaterThanOne = 0
numTrain = 0
numTest = 0

for row in reader:
    if rowCount < censored_train_cutoff:
        userDict[row[4]] += 1
    else:
        break
    rowCount += 1

f.seek(0)

rowCount = 0
for row in reader:
    if row[0] != "mid":
        feature_vector = []

        positives = 0
        negatives = 0

        these_words = set([])
        for w in row[2].split(" "):
            these_words.add(w)

            if w in pos_set:
                positives += 1

            if w in neg_set:
                negatives += 1

        for word in feature_words:
            if word in these_words:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

        # tweet has an image or not
        if row[5] == "1":
            feature_vector.append(1)
        else:
            feature_vector.append(0)

        # gender of the user
        if row[9] == "m":
            feature_vector.append(1)
        else:
            feature_vector.append(0)

        # province of the user
        for province in provinces:
            if int(row[8]) == province:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

        # whether the message is a retweet
        if row[3] == "":
            feature_vector.append(0)
        else:
            feature_vector.append(1)

        # total number of tweets of the user that were censored
        feature_vector.append(userDict[row[4]])

        dateFormat = "%Y-%m-%d %H:%M:%S.%f"
        dateFormat2 = "%Y-%m-%d %H:%M:%S"

        try:
            deletedTime = datetime.datetime.strptime(row[7], dateFormat)
        except:
            try:
                deletedTime = datetime.datetime.strptime(row[7], dateFormat2)
            except:
                continue

        try:
            postedTime = datetime.datetime.strptime(row[6], dateFormat)
        except:
            try:
                postedTime = datetime.datetime.strptime(row[6], dateFormat2)
            except:
                continue

        delta = deletedTime - postedTime
        days = delta.total_seconds() / float(60 * 60 * 24)
        feature_vector.append(days)
        if days > 1:
            greaterThanOne += 1

        if addedCount < censored_train_cutoff:
            train_matrix.append(feature_vector)
            numTrain += 1
        else:
            test_matrix.append(feature_vector)
            numTest += 1

        addedCount += 1

    rowCount += 1


print("numTrain" + str(numTrain))
print("numTest" + str(numTest))
print("cutoff" + str(censored_train_cutoff))
print("addedCount" + str(addedCount))
print("greaterThanOne" + str(greaterThanOne))
print("building train/test X,Y")

train_X = []
test_X = []
train_Y = []
test_Y = []

random.shuffle(train_matrix)

for element in train_matrix:
    train_Y.append(element[len(element) - 1])
    train_X.append(element[0:len(element) - 1])

for element in test_matrix:
    test_Y.append(element[len(element) - 1])
    test_X.append(element[0:len(element) - 1])

print("Training Linear Regression with Regularization")
clf = linear_model.Ridge(alpha=10)
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)

diffs = []

for i in range(len(preds)):
    if i < 50:
        print()
        str(preds[i]) + " : " + str(test_Y[i])
    diffs.append(abs(preds[i] - test_Y[i]))

print("Average Absolute Error: " + str(np.mean(diffs)))
print("Standard Deviation: " + str(np.std(diffs)))
print("Training SVR")
clf = SVR()
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)

diffs = []

for i in range(len(preds)):
    if i < 50:
        print()
        str(preds[i]) + " : " + str(test_Y[i])
    diffs.append(abs(preds[i] - test_Y[i]))

print("Average Absolute Error: " + str(np.mean(diffs)))
print("Standard Deviation: " + str(np.std(diffs)))
