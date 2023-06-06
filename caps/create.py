#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/6/6 18:56
# @File     : create.py
# @Project  : final-pj-weibo
import csv
import datetime
import json


def import_exceptions():
    """
    the function imports the KeyError exceptions saved in a txt file
    """
    with open('../data/exceptions.txt', 'r') as infile:
        data = list(csv.reader(infile))[0]
    return list(map(int, data))


def test_key_errors(errors):
    """
    the function creates a list of keys that are valid to be valed and checks whether they give a KeyError
    """

    key_list = []

    for i in range(31963):
        if i + 1 not in errors:
            try:
                var = data[str(i + 1)]
                key_list.append(str(i + 1))
            except KeyError:
                print("key error at ", i + 1)

    return key_list


def change_dict_keys(data, key_list):
    """
    changes the dictionary keys, which contain the original json file names, to ordered key integers
    """

    new_data = {}

    for i in range(len(data) - 1):
        new_data[i] = data[key_list[i]]

    with open('../data/verdicts_ordered_keys.json', 'w') as f:
        json.dump(new_data, f)

    return new_data


def save_to_csv(new_data):
    """
    this function converts the json file into csv file
    """

    with open('../data/verdicts.csv', 'w') as f:
        fieldnames = ['Unique_ID', 'Word_ID', 'Word', 'Time_UNIX', 'Time_year', 'Time_month', 'Time_day',
                      'week_aggregator', 'Verdict']

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        counter = 0
        undefined_counter = 0

        for i in range(len(new_data)):
            index = str(i)
            for key, value in new_data[index]['verdicts'].items():

                '''
                The below row variable saves to each row the indicated data of the given key-value (time-verdict) observation. As the timestamps have been saved not as objects but as strings, they need to be converted back to float objects.
                '''

                '''
                The year, month and day variables are only to keep the date convention. We use the week group aggergator in our analysis.
                '''

                if value != 1:
                    dtime = datetime.datetime.fromtimestamp(float(key))
                    year = dtime.strftime('%Y')
                    month = dtime.strftime('%m')
                    week = str(int(dtime.strftime('%U')) + (int(year[3]) - 2) * 52)
                    day = dtime.strftime('%d')

                    row = {'Unique_ID': counter, 'Word_ID': i, 'Word': new_data[index]['url'][25:],
                           'Time_UNIX': float(key), 'Time_year': year, 'Time_month': month, 'Time_day': day,
                           'week_aggregator': week, 'Verdict': int(value)}

                    counter += 1

                    writer.writerow(row)
                else:
                    '''
                    if the verdict value was one, meaning undefined, then registry of such key-value pair was ommitted 
                    '''
                    undefined_counter += 1
                    pass

    return undefined_counter, counter


if __name__ == '__main__':
    with open('../data/verdicts.json', 'r') as infile:
        data = json.load(infile)

    # errors = import_exceptions()
    # key_list = test_key_errors(errors)
    # new_data = change_dict_keys(data, key_list)

    with open('../data/verdicts_ordered_keys.json') as f:
        new_data = json.load(f)

    undefined_counter, counter = save_to_csv(new_data)

    print("The number of observations with undefined outcomes is: ", undefined_counter)
    print("The number of total useful observations is: ", counter - undefined_counter)
