#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 20:31
# @File     : classify.py
# @Project  : final-pj-weibo
import csv
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB

vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18)
provinces = [11, 82, 54]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

mi_read = open(mi_infile := "../../data/raw_tbd/mi_results.csv", 'rb')
reader = csv.reader(mi_read, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
rowCount = 0
feature_words = []

print("loading sentiment sets")
pos = open('pos.txt', 'rb')
neg = open('neg.txt', 'rb')
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
        
    if rowCount < 2500:
        feature_words.append(row[0])
    else:
        break
    rowCount += 1

rowCount = 0
mi_read = open(mi_infile := "../../data/raw_tbd/diff_word_frequencies.csv", 'rb')
reader = csv.reader(mi_read, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)

for row in reader:
    if rowCount < 0:
        feature_words.append(row[0])
    else:
        break
    rowCount += 1


print(len(feature_words))
train_matrix = []
test_matrix = []

print("adding censored data")
f = open(infile := "../../data/raw_tbd/all_censored.csv", 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
censored_train_cutoff = 85802 - (85802 / float(10))  # /float(7)
userDict = defaultdict(int)
rowCount = 0

for row in reader:
    if rowCount < censored_train_cutoff:  # only record for training examples
        userDict[row[4]] += 1
    else:
        break
    rowCount += 1

f.seek(0)

rowCount = 0
mCount = 0
tCount = 0
iCount = 0
addedCount = 0
posCount = 0
negCount = 0
retweetCount = 0
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
            iCount += 1
        else:
            feature_vector.append(0)

        # gender of the user
        if row[9] == "m":
            feature_vector.append(1)
            mCount += 1
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
            retweetCount += 1

        # total number of tweets of the user that were censored
        feature_vector.append(userDict[row[4]])

        # carries the class label - deleted afterwards
        feature_vector.append(1)
        if rowCount < censored_train_cutoff:
            train_matrix.append(feature_vector)
        else:
            test_matrix.append(feature_vector)

        addedCount += 1

    rowCount += 1
# print() str(rowCount) + "\r"
# sys.stdout.flush()

print("mCount" + str(mCount))
print("tCount" + str(tCount))
print("iCount" + str(iCount))
print("addedCount" + str(addedCount))
print("retweetCount" + str(retweetCount))
print("posCount" + str(posCount))
print("negCount" + str(negCount))
print("adding uncensored data")

uncensored = "../../data/raw_tbd/full_uncensored_sample_2.csv"
uc = open(uncensored, 'rb')
reader = csv.reader(uc, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
uncensored_train_cutoff = 93631 - (93631 / float(10))
rowCount = 0
added = 0
retweetCount = 0
posCount = 0
negCount = 0

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

        if row[5] == "1":
            feature_vector.append(1)
        else:
            feature_vector.append(0)

        if row[9] == "m":
            feature_vector.append(1)
        else:
            feature_vector.append(0)

        for province in provinces:
            if int(row[8]) == province:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

        if row[3] == "":
            feature_vector.append(0)
        else:
            feature_vector.append(1)
            retweetCount += 1

        feature_vector.append(userDict[row[4]])

        feature_vector.append(0)
        if rowCount < uncensored_train_cutoff:
            train_matrix.append(feature_vector)
        else:
            test_matrix.append(feature_vector)

        added += 1

    rowCount += 1
# print() str(rowCount) + "\r"
# sys.stdout.flush()

print("Added: " + str(added))
print("retweetCount" + str(retweetCount))
print("posCount" + str(posCount))
print("negCount" + str(negCount))
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

print("Training SGD")

all_classes = np.array([0, 1])
sgd = SGDClassifier(penalty='elasticnet', loss='log', class_weight={0: 0.3, 1: 0.7})

clf = sgd

train_X_sets = []
train_Y_sets = []
partitions = 50

for i in range(1, partitions + 1):
    this_set = train_X[(i - 1) * len(train_X) / partitions:i * len(train_X) / partitions]
    this_set_y = train_Y[(i - 1) * len(train_X) / partitions:i * len(train_X) / partitions]
    train_X_sets.append(this_set)
    train_Y_sets.append(this_set_y)

print(len(train_X_sets))
print(len(train_Y_sets))

# train_X_1 = train_X[0:len(train_X)/10]
# train_X_2 = train_X[len(train_X)/10:2*len(train_X)/10]
# train_X_3 = train_X[2*len(train_X)/10:3*len(train_X)/10]
# train_X_4 = train_X[3*len(train_X)/10:4*len(train_X)/10]
# train_X_5 = train_X[4*len(train_X)/10:5*len(train_X)/10]
# train_X_6 = train_X[5*len(train_X)/10:6*len(train_X)/10]
# train_X_7 = train_X[6*len(train_X)/10:7*len(train_X)/10]
# train_X_8 = train_X[7*len(train_X)/10:8*len(train_X)/10]
# train_X_9 = train_X[8*len(train_X)/10:9*len(train_X)/10]
# train_X_10 = train_X[9*len(train_X)/10:10*len(train_X)/10]


# train_Y_1 = train_Y[0:len(train_X)/10]
# train_Y_2 = train_Y[len(train_X)/10:2*len(train_X)/10]
# train_Y_3 = train_Y[2*len(train_X)/10:3*len(train_X)/10]
# train_Y_4 = train_Y[3*len(train_X)/10:4*len(train_X)/10]
# train_Y_5 = train_Y[4*len(train_X)/10:5*len(train_X)/10]
# train_Y_6 = train_Y[5*len(train_X)/10:6*len(train_X)/10]
# train_Y_7 = train_Y[6*len(train_X)/10:7*len(train_X)/10]
# train_Y_8 = train_Y[7*len(train_X)/10:8*len(train_X)/10]
# train_Y_9 = train_Y[8*len(train_X)/10:9*len(train_X)/10]
# train_Y_10 = train_Y[9*len(train_X)/10:10*len(train_X)/10]

clf.partial_fit(train_X_sets[0], train_Y_sets[0], all_classes)

print("Results - partial1:")
preds = clf.predict(test_X)

if len(preds) != len(test_Y):
    print("lengths don't match")

correct = 0
incorrect = 0
correct_censored = 0
incorrect_censored = 0
correct_uncensored = 0
incorrect_uncensored = 0

for i in range(len(preds)):
    if preds[i] == test_Y[i]:
        correct += 1
    else:
        incorrect += 1

    if test_Y[i] == 1 and preds[i] == 1:
        correct_censored += 1
    if test_Y[i] == 1 and preds[i] == 0:
        incorrect_censored += 1
    if test_Y[i] == 0 and preds[i] == 0:
        correct_uncensored += 1
    if test_Y[i] == 0 and preds[i] == 1:
        incorrect_uncensored += 1

print("Total Accuracy: " + str(correct / float(correct + incorrect)))
print("Censored Accuracy: " + str(correct_censored / float(correct_censored + incorrect_censored)))
print("Uncensored Accuracy: " + str(correct_uncensored / float(correct_uncensored + incorrect_uncensored)))


for i in range(1, partitions):
    clf.partial_fit(train_X_sets[i], train_Y_sets[i], all_classes)

# clf.partial_fit(train_X_2, train_Y_2, all_classes)
# clf.partial_fit(train_X_3, train_Y_3, all_classes)
# clf.partial_fit(train_X_4, train_Y_4, all_classes)
# clf.partial_fit(train_X_5, train_Y_5, all_classes)
# clf.partial_fit(train_X_6, train_Y_6, all_classes)
# clf.partial_fit(train_X_7, train_Y_7, all_classes)
# clf.partial_fit(train_X_8, train_Y_8, all_classes)
# clf.partial_fit(train_X_9, train_Y_9, all_classes)
# clf.partial_fit(train_X_10, train_Y_10, all_classes)

print("Results - partial" + str(partitions) + ":")

preds = clf.predict(test_X)

if len(preds) != len(test_Y):
    print("lengths don't match")


correct = 0
incorrect = 0
correct_censored = 0
incorrect_censored = 0
correct_uncensored = 0
incorrect_uncensored = 0

for i in range(len(preds)):
    if preds[i] == test_Y[i]:
        correct += 1
    else:
        incorrect += 1

    if test_Y[i] == 1 and preds[i] == 1:
        correct_censored += 1
    if test_Y[i] == 1 and preds[i] == 0:
        incorrect_censored += 1
    if test_Y[i] == 0 and preds[i] == 0:
        correct_uncensored += 1
    if test_Y[i] == 0 and preds[i] == 1:
        incorrect_uncensored += 1

print("Total Accuracy: " + str(correct / float(correct + incorrect)))
print("Censored Accuracy: " + str(correct_censored / float(correct_censored + incorrect_censored)))
print("Uncensored Accuracy: " + str(correct_uncensored / float(correct_uncensored + incorrect_uncensored)))
print("Training Logistic Regression L1")

# parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'tol':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
lr = LogisticRegression(penalty='l1', class_weight={0: 0.2, 1: 0.8})
# clf = RFECV(lr)
clf = lr
# clf = grid_search.GridSearchCV(lr, parameters)
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)

# ROC CURVE
probs = clf.predict_proba(test_X)
fpr_lr_l1, tpr_lr_l1, thresholds_lr_l1 = metrics.roc_curve(test_Y, probs.T[1], pos_label=1)
roc_auc_lr_l1 = metrics.auc(fpr_lr_l1, tpr_lr_l1)

print("AUC L1: " + str(roc_auc_lr_l1))

if len(preds) != len(test_Y):
    print("lengths don't match")


false_positives = open('../../data/raw_tbd/false_positives.csv', 'wb')
false_negatives = open('../../data/raw_tbd/falses_negatives.csv', 'wb')
true_positives = open('../../data/raw_tbd/true_positives.csv', 'wb')
true_negatives = open('../../data/raw_tbd/true_negatives.csv', 'wb')

fp_writer = csv.writer(false_positives, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
fn_writer = csv.writer(false_negatives, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
tp_writer = csv.writer(true_positives, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
tn_writer = csv.writer(true_negatives, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)

correct = 0
incorrect = 0
correct_censored = 0
incorrect_censored = 0
correct_uncensored = 0
incorrect_uncensored = 0

for i in range(len(preds)):
    if preds[i] == test_Y[i]:
        correct += 1
    else:
        incorrect += 1

    if test_Y[i] == 1 and preds[i] == 1:
        correct_censored += 1
        string = ""
        for w_index in range(len(feature_words)):
            if train_X[i][w_index] == 1:
                string = string + feature_words[w_index] + " "

        tp_writer.writerow([string])
    if test_Y[i] == 1 and preds[i] == 0:
        incorrect_censored += 1
        string = ""
        for w_index in range(len(feature_words)):
            if train_X[i][w_index] == 1:
                string = string + feature_words[w_index] + " "

        fn_writer.writerow([string])
    if test_Y[i] == 0 and preds[i] == 0:
        correct_uncensored += 1
        string = ""
        for w_index in range(len(feature_words)):
            if train_X[i][w_index] == 1:
                string = string + feature_words[w_index] + " "

        tn_writer.writerow([string])
    if test_Y[i] == 0 and preds[i] == 1:
        incorrect_uncensored += 1
        string = ""
        for w_index in range(len(feature_words)):
            if train_X[i][w_index] == 1:
                string = string + feature_words[w_index] + " "

        fp_writer.writerow([string])

print("Total Accuracy: " + str(correct / float(correct + incorrect)))
print("Censored Accuracy: " + str(correct_censored / float(correct_censored + incorrect_censored)))
print("Uncensored Accuracy: " + str(correct_uncensored / float(correct_uncensored + incorrect_uncensored)))
print("Rankings: ")
print(clf.ranking_) 

print("Training Logistic Regression L2")

# class_weights = [{0:}]
# parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'tol':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
lr = LogisticRegression(penalty='l2', class_weight={0: 0.2, 1: 0.8})
# clf = RFECV(lr)
clf = lr
# clf = grid_search.GridSearchCV(lr, parameters)
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)

# ROC CURVE:
probs = clf.predict_proba(test_X)
fpr_lr_l2, tpr_lr_l2, thresholds_lr_l2 = metrics.roc_curve(test_Y, probs.T[1], pos_label=1)
roc_auc_lr_l2 = metrics.auc(fpr_lr_l2, tpr_lr_l2)

print("AUC L2: " + str(roc_auc_lr_l2))
print(len(params := clf.coef_[0]))


paramDict = defaultdict(float)
paramCount = 0
for i in range(len(feature_words)):
    paramDict[feature_words[i]] = params[i]
    paramCount += 1

of = open("../../data/raw_tbd/l2_coefficients.csv", 'wb')
writer = csv.writer(of, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
otherParams = ["image", "gender", "beijing", "macau", "gansu" "retweet", "same_user_censor_count"]
for p in otherParams:
    paramDict[p] = params[paramCount]
    paramCount += 1

for w in sorted(paramDict, key=paramDict.get, reverse=True):
    writer.writerow([w, paramDict[w]])

if len(preds) != len(test_Y):
    print("lengths don't match")
    

correct = 0
incorrect = 0
correct_censored = 0
incorrect_censored = 0
correct_uncensored = 0
incorrect_uncensored = 0

for i in range(len(preds)):
    if preds[i] == test_Y[i]:
        correct += 1
    else:
        incorrect += 1

    if test_Y[i] == 1 and preds[i] == 1:
        correct_censored += 1
    if test_Y[i] == 1 and preds[i] == 0:
        incorrect_censored += 1
    if test_Y[i] == 0 and preds[i] == 0:
        correct_uncensored += 1
    if test_Y[i] == 0 and preds[i] == 1:
        incorrect_uncensored += 1

print("Total Accuracy: " + str(correct / float(correct + incorrect)))
print("Censored Accuracy: " + str(correct_censored / float(correct_censored + incorrect_censored)))
print("Uncensored Accuracy: " + str(correct_uncensored / float(correct_uncensored + incorrect_uncensored)))
print("Rankings: ")
print(clf.ranking_)
print("Training Bernoulli Naive Bayes")

clf = BernoulliNB(alpha=1.0)
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)
probs = clf.predict_proba(test_X)
fpr_lr_nb, tpr_lr_nb, thresholds_lr_nb = metrics.roc_curve(test_Y, probs.T[1], pos_label=1)
roc_auc_lr_nb = metrics.auc(fpr_lr_nb, tpr_lr_nb)

print("AUC nb: " + str(roc_auc_lr_nb))

correct = 0
incorrect = 0
correct_censored = 0
incorrect_censored = 0
correct_uncensored = 0
incorrect_uncensored = 0

for i in range(len(preds)):
    if preds[i] == test_Y[i]:
        correct += 1
    else:
        incorrect += 1

    if test_Y[i] == 1 and preds[i] == 1:
        correct_censored += 1
    if test_Y[i] == 1 and preds[i] == 0:
        incorrect_censored += 1
    if test_Y[i] == 0 and preds[i] == 0:
        correct_uncensored += 1
    if test_Y[i] == 0 and preds[i] == 1:
        incorrect_uncensored += 1

print("Total Accuracy: " + str(correct / float(correct + incorrect)))
print("Censored Accuracy: " + str(correct_censored / float(correct_censored + incorrect_censored)))
print("Uncensored Accuracy: " + str(correct_uncensored / float(correct_uncensored + incorrect_uncensored)))

###########################
# plot ROC curve

plt.figure()
plt.plot(fpr_lr_l1, tpr_lr_l1, color='red', label='ROC curve (area = %0.2f)' % roc_auc_lr_l1)
plt.plot(fpr_lr_l2, tpr_lr_l2, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_lr_l2)
plt.plot(fpr_lr_nb, tpr_lr_nb, color='yellow', label='ROC curve (area = %0.2f)' % roc_auc_lr_nb)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.show()

###########################
