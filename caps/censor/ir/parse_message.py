#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 21:03
# @File     : parse_message.py
# @Project  : final-pj-weibo
import csv
import os
import sys
import pynlpir


def main(argv):
    csv.field_size_limit(sys.maxsize)

    pynlpir.open()
    # load sogou dictionaries
    for fn in os.listdir(root := "../../data/raw_tbd/weibo-data/"):
        n = pynlpir.nlpir.ImportUserDict(os.path.join(root, fn))
        print("num imported: " + str(n))

    # for fi in files:
    fi = "week" + str(argv[0]) + ".csv"
    print(fi)
    # for fi in files:
    infile = "./../" + fi
    a = fi.split('.')
    outfile = "./../parsed/" + a[0] + "parsed.csv"

    f = open(infile, 'rb')
    of = open(outfile, 'wb')
    reader = csv.reader(f, delimiter=",")
    writer = csv.writer(of, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
    count = 0
    total = sum(1 for _ in reader)
    print(total)
    f.seek(0)
    errors = 0
    unbounderrors = 0

    for row in reader:
        mid = row[0]
        message = row[6]
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

        if row[10] != "":
            censored = 1
        else:
            censored = 0

        writer.writerow([mid, censored, mString.encode("utf-8")])

        # progress
        if count % 1000 == 0:
            print(str(count) + "/" + str(total) + "\r")
            sys.stdout.flush()
        count += 1

    print("count: " + str(count))
    print("errors: " + str(errors))
    print("unbounderrors: " + str(unbounderrors))


if __name__ == "__main__":
    main(sys.argv[1:])
