#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 22:39
# @File     : sample.py
# @Project  : final-pj-weibo
import csv
import os
import sys
import pynlpir


def main():
    csv.field_size_limit(sys.maxsize)
    pynlpir.open()
    # load sogou dictionaries
    for fn in os.listdir(root := "../../data/raw_tbd/weibo-data/"):
        n = pynlpir.nlpir.ImportUserDict(os.path.join(root, fn))
        print("num imported: " + str(n))

    # write out parsed words
    files = ["week" + str(i) + ".csv" for i in range(1, 53)]

    print("loading user data into map")
    userDict = dict()
    userdata = "../../data/raw_tbd/userdata.csv"
    with open(userdata, 'rb') as users:
        reader = csv.reader(users, delimiter=",")
        for row in reader:
            userDict[row[0]] = [row[1], row[2], row[3]]

    outfile = "../../data/raw_tbd/full_uncensored_sample.csv"
    addedCount = 0

    for fi in files:
        # fi = "week" + str(argv[0]) + ".csv"
        print(fi)
        # for fi in files:
        infile = "../../data/raw_tbd/" + fi
        a = fi.split('.')

        f = open(infile, 'rb')
        of = open(outfile, 'a')
        reader = csv.reader(f, delimiter=",")
        writer = csv.writer(of, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
        count = 0
        total = sum(1 for row in reader)
        print(total)
        f.seek(0)
        errors = 0
        unbounderrors = 0

        for row in reader:
            if count % 2421 == 0 and row[10] == "":
                mid = row[0]
                message = row[6]
                censored = 0
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
                    mString += segment[0]
                    mString += " "

                try:
                    d = userDict[row[2]]
                except KeyError:
                    # print("no key for userid " + row[2]
                    continue
                except ValueError:
                    print("ValueError")
                    continue

                writer.writerow(
                    [mid, censored, mString.encode("utf-8"), row[1], row[2], row[5], row[8], row[9], d[0], d[1], d[2]])
                addedCount += 1
            # progress
            if count % 1000 == 0:
                print(str(count) + "/" + str(total) + "\r")
                sys.stdout.flush()
            count += 1

        print("addedCount: " + str(addedCount))
        print("count: " + str(count))
        print("errors: " + str(errors))
        print("unbounderrors: " + str(unbounderrors))


if __name__ == "__main__":
    main()
