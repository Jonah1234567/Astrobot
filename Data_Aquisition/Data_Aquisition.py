import numpy as np
from numpy import unique
from numpy import argmax
import math
from pandas import read_csv
import pandas
from matplotlib import pyplot
from baseball_scraper import statcast
from baseball_scraper import playerid_lookup, statcast_pitcher, pitching_stats, batting_stats_range
import os.path
from os import path


def scrape_pitch_data(sdate, edate):
    # scrapes data of all pitchers between two dates
    data_path = 'Data\\' + edate + ".csv"
    for i in range(3, 11):
        for j in range(1, 32):

            month = str(i)
            day = str(j)
            if i < 10:
                month = str(0) + str(i)
            if j < 10:
                day = str(0) + str(j)

            old_data = 'Data\\' "_2021" + "-" + month + "-" + day + ".csv"

            if path.isfile(old_data) and old_data == data_path:
                return dataPipe3(data_path)

            elif path.isfile(old_data) and old_data != data_path:
                os.rename(old_data, data_path)
                df = pandas.DataFrame(statcast(start_dt=sdate, end_dt=edate))
                df.iloc[:, 1:].to_csv(data_path, index=True)
                return dataPipe3(data_path)

    df = pandas.DataFrame(statcast(start_dt=sdate, end_dt=edate))
    df.iloc[:, 1:].to_csv(data_path, index=True)
    print(df.iloc[:, 1:].shape)
    return dataPipe3(data_path)


def scrape_pitch_data_player(start_date, end_date, first_name, last_name):
    # scrapes data of a single pitcher between two dates
    data_path = 'Data\\' + first_name + "_" + last_name + "_" + end_date + ".csv"
    for i in range(3, 11):
        for j in range(1, 32):

            month = str(i)
            day = str(j)
            if i < 10:
                month = str(0) + str(i)
            if j < 10:
                day = str(0) + str(j)

            old_data = 'Data\\' + first_name + "_" + last_name + "_2021" + "-" + month + "-" + day + ".csv"

            if path.isfile(old_data) and old_data == data_path:
                return dataPipe3(data_path)

            elif path.isfile(old_data) and old_data != data_path:
                os.rename(old_data, data_path)
                info = pandas.DataFrame(playerid_lookup(last_name, first_name))
                print(info.iloc[0, 2])
                df = pandas.DataFrame(statcast_pitcher(start_date, end_date, int(info.iloc[0, 2])))
                df.to_csv(data_path, index=True)
                return dataPipe3(data_path)

    info = pandas.DataFrame(playerid_lookup(last_name, first_name))
    print(info.iloc[0, 2])
    df = pandas.DataFrame(statcast_pitcher(start_date, end_date, int(info.iloc[0, 2])))
    df.to_csv(data_path, index=True)
    return dataPipe3(data_path)


def scrape_pitch_data_player_Categorical(start_date, end_date, first_name, last_name):
    # scrapes data of a pitcher and makes it categorical (int) between two dates
    data_path = 'Data\\' + first_name + "_" + last_name + "_" + end_date + ".csv"
    for i in range(3, 11):
        for j in range(1, 32):

            month = str(i)
            day = str(j)
            if i < 10:
                month = str(0) + str(i)
            if j < 10:
                day = str(0) + str(j)

            old_data = 'Data\\' + first_name + "_" + last_name + "_2021" + "-" + month + "-" + day + ".csv"

            if path.isfile(old_data) and old_data == data_path:
                return dataPipe3_Categorical(data_path)

            elif path.isfile(old_data) and old_data != data_path:
                os.rename(old_data, data_path)
                info = pandas.DataFrame(playerid_lookup(last_name, first_name))
                print(info.iloc[0, 2])
                df = pandas.DataFrame(statcast_pitcher(start_date, end_date, int(info.iloc[0, 2])))
                df.to_csv(data_path, index=True)
                return dataPipe3_Categorical(data_path)

    info = pandas.DataFrame(playerid_lookup(last_name, first_name))
    print(info.iloc[0, 2])
    df = pandas.DataFrame(statcast_pitcher(start_date, end_date, int(info.iloc[0, 2])))
    df.to_csv(data_path, index=True)
    return dataPipe3_Categorical(data_path)


def scrape_pitch_data_player_binary(start_date, end_date, first_name, last_name):
    #scrapes data of a pitcher and makes it binary (fastball, offspeed) between two dates
    data_path = 'Data\\' + first_name + "_" + last_name + "_" + end_date + ".csv"
    for i in range(3, 11):
        for j in range(1, 32):

            month = str(i)
            day = str(j)
            if i < 10:
                month = str(0) + str(i)
            if j < 10:
                day = str(0) + str(j)

            old_data = 'Data\\' + first_name + "_" + last_name + "_2021" + "-" + month + "-" + day + ".csv"

            if path.isfile(old_data) and old_data == data_path:
                return dataPipeBinary(data_path)

            elif path.isfile(old_data) and old_data != data_path:
                os.rename(old_data, data_path)
                info = pandas.DataFrame(playerid_lookup(last_name, first_name))
                print(info.iloc[0, 2])
                df = pandas.DataFrame(statcast_pitcher(start_date, end_date, int(info.iloc[0, 2])))
                df.to_csv(data_path, index=True)
                return dataPipe3(data_path)

    info = pandas.DataFrame(playerid_lookup(last_name, first_name))
    print(info.iloc[0, 2])
    df = pandas.DataFrame(statcast_pitcher(start_date, end_date, int(info.iloc[0, 2])))
    df.to_csv(data_path, index=True)
    return dataPipeBinary(data_path)


def dataPipe3(path):
    # takes raw data from baseball stat scraper and turns it into a usable numpy array
    # seperates pitches into 3 pitches: fastball, breaking ball, changeup

    dataframe = read_csv(path, header=0)
    pitches = dataframe.iloc[0:, 1]
    pitches_List = ["Fastball", "Breaking Ball", "Offspeed", "Other"]
    counter = 0

    dl_Length = len(pitches)
    num_Pitches = len(pitches_List)

    dataset = np.zeros([dl_Length, num_Pitches * 8 + 23])
    x_Length, y_Length = np.shape(dataset)
    for i in range(len(pitches)):
        if pitches[i] == "FF" or pitches[i] == "FT" or pitches[i] == "FS" or pitches[i] == "FC" or pitches[i] == "SI":
            # fastball
            dataset[i, y_Length - 1] = 0
        elif pitches[i] == "CU" or pitches[i] == "SL" or pitches[i] == "KC" or pitches[i] == "EP":  # breaking ball
            dataset[i, y_Length - 1] = 1
        elif pitches[i] == "CH" or pitches[i] == "SF" or pitches[i] == "SC" or pitches[i] == "FO" or pitches[i] == "KN":
            # offspeed
            dataset[i, y_Length - 1] = 2
        else:
            dataset[i, y_Length - 1] = 3

    for i in range(dl_Length):
        # date
        dataset[i, 0] = int(str(dataframe.iloc[i, 2]).replace("-", ""))
        # batter ID
        dataset[i, 1] = int(dataframe.iloc[i, 7])
        # pitcher ID
        dataset[i, 2] = int(dataframe.iloc[i, 8])

        # R is 0 and L is one
        if dataframe.iloc[i, 17] == "R":
            dataset[i, 3] = 0
        elif dataframe.iloc[i, 17] == "L":
            dataset[i, 3] = 1

        # R is 0 and L is one
        if dataframe.iloc[i, 18] == "R":
            dataset[i, 4] = 0
        elif dataframe.iloc[i, 18] == "L":
            dataset[i, 4] = 1

        # R is 0 and L is one
        if dataframe.iloc[i, 19] == "R":
            dataset[i, 5] = 0
        elif dataframe.iloc[i, 19] == "L":
            dataset[i, 5] = 1

        # Ball
        dataset[i, 6] = int(dataframe.iloc[i, 25])
        # Strike
        dataset[i, 7] = int(dataframe.iloc[i, 26])

        # On base
        # 3B
        if math.isnan(dataframe.iloc[i, 32]):

            dataset[i, 8] = 0
        else:
            dataset[i, 8] = 1

        # 2B
        if math.isnan(dataframe.iloc[i, 32]):
            dataset[i, 9] = 0
        else:
            dataset[i, 9] = 1
        # 1B
        if math.isnan(dataframe.iloc[i, 32]):
            dataset[i, 10] = 0
        else:
            dataset[i, 10] = 1

        # innings
        dataset[i, 11] = int(dataframe.iloc[i, 36])

        # outs
        dataset[i, 12] = int(dataframe.iloc[i, 35])

        # Catcher
        if math.isnan(dataframe.iloc[i, 61]):
            dataset[i, 13] = 0
        else:
            dataset[i, 13] = int(dataframe.iloc[i, 61])

        # Score
        # bat score
        dataset[i, 14] = int(dataframe.iloc[i, 82])
        # Field Score
        dataset[i, 15] = int(dataframe.iloc[i, 83])

        # Year/Career Pitches number
        dataset[i, 16] = dl_Length - 1 - i

        dataset[dl_Length - 1, 17] = 0
        dataset[dl_Length - 1, 18] = 0
        dataset[dl_Length - 1, 19] = 0
        dataset[dl_Length - 1, 20:y_Length - 4] = 0
        dataset[dl_Length - 1, y_Length - 3] = -1
        dataset[dl_Length - 1, y_Length - 2] = 1
        dataset[dl_Length - 2, y_Length - 2] = 1

    # game, out and inning pitch count
    for i in range(dl_Length - 2, -1, -1):
        # streak type
        dataset[i, y_Length - 3] = dataset[i + 1, y_Length - 1]
        # game
        if dataset[i, 0] != dataset[i + 1, 0]:
            dataset[i, 17] = 0
        else:
            dataset[i, 17] = dataset[i + 1, 17] + 1

        # inning
        if dataset[i, 11] != dataset[i + 1, 11]:
            dataset[i, 18] = 0
        else:
            dataset[i, 18] = dataset[i + 1, 18] + 1

        # out
        if dataset[i, 1] != dataset[i + 1, 1]:
            dataset[i, 19] = 0
        else:
            dataset[i, 19] = dataset[i + 1, 19] + 1

        # year/career pitch tracker
        dataset[i, 20:20 + num_Pitches] = dataset[i + 1, 20:20 + num_Pitches]
        dataset[i, 20 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 16] == 0:
            dataset[i, 20 + num_Pitches:20 + num_Pitches * 2] = 0
        else:
            dataset[i, 20 + num_Pitches:20 + num_Pitches * 2] = dataset[i, 20:20 + num_Pitches] / dataset[i, 16]

        # game pitch tracker
        if dataset[i + 1, 0] != dataset[i, 0]:
            dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] = 0
        else:
            dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] = dataset[i + 1,
                                                                    20 + num_Pitches * 2:20 + num_Pitches * 3]
            dataset[i, 20 + num_Pitches * 2 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 17] == 0:
            dataset[i, 20 + num_Pitches * 3:20 + num_Pitches * 4] = 0
        else:
            dataset[i, 20 + num_Pitches * 3:20 + num_Pitches * 4] = dataset[i,
                                                                    20 + num_Pitches * 2:20 + num_Pitches * 3] / \
                                                                    dataset[i, 17]

        # inning pitch tracker
        if dataset[i + 1, 11] != dataset[i, 11]:
            dataset[i, 20 + num_Pitches * 4:20 + num_Pitches * 5] = 0
        else:
            dataset[i, 20 + num_Pitches * 4:20 + num_Pitches * 5] = dataset[i + 1,
                                                                    20 + num_Pitches * 4:20 + num_Pitches * 5]
            dataset[i, 20 + num_Pitches * 4 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 18] == 0:
            dataset[i, 20 + num_Pitches * 5:20 + num_Pitches * 6] = 0
        else:
            dataset[i, 20 + num_Pitches * 5:20 + num_Pitches * 6] = dataset[i,
                                                                    20 + num_Pitches * 4:20 + num_Pitches * 5] / \
                                                                    dataset[i, 18]

        # out pitch tracker

        if dataset[i + 1, 1] != dataset[i, 1]:
            dataset[i, 20 + num_Pitches * 6:20 + num_Pitches * 7] = 0
        else:
            dataset[i, 20 + num_Pitches * 6:20 + num_Pitches * 7] = dataset[i + 1,
                                                                    20 + num_Pitches * 6:20 + num_Pitches * 7]
            dataset[i, 20 + num_Pitches * 6 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 19] == 0:
            dataset[i, 20 + num_Pitches * 7:20 + num_Pitches * 8] = 0
        else:
            dataset[i, 20 + num_Pitches * 7:20 + num_Pitches * 8] = dataset[i,
                                                                    20 + num_Pitches * 6:20 + num_Pitches * 7] / \
                                                                    dataset[i, 19]

            # streak length
    for i in range(dl_Length - 3, -1, -1):
        if dataset[i + 1, y_Length - 1] == dataset[i + 2, y_Length - 1]:
            dataset[i, y_Length - 2] = dataset[i + 1, y_Length - 2] + 1

    return dataset


def dataPipe3_Categorical(path):
    # takes raw data from baseball stat scraper and turns it into a usable numpy array, turns everything into an int
    # seperates pitches into 3 pitches: fastball, breaking ball, changeup
    dataframe = read_csv(path, header=0)
    pitches = dataframe.iloc[0:, 1]
    pitches_List = ["Fastball", "Breaking Ball", "Offspeed", "Other"]
    counter = 0

    dl_Length = len(pitches)
    num_Pitches = len(pitches_List)

    dataset = np.zeros([dl_Length, num_Pitches * 8 + 23])
    x_Length, y_Length = np.shape(dataset)
    for i in range(len(pitches)):
        if pitches[i] == "FF" or pitches[i] == "FT" or pitches[i] == "FS" or pitches[i] == "FC" or pitches[i] == "SI":
            # fastball
            dataset[i, y_Length - 1] = 0
        elif pitches[i] == "CU" or pitches[i] == "SL" or pitches[i] == "KC" or pitches[i] == "EP":  # breaking ball
            dataset[i, y_Length - 1] = 1
        elif pitches[i] == "CH" or pitches[i] == "SF" or pitches[i] == "SC" or pitches[i] == "FO" or pitches[i] == "KN":
            # offspeed
            dataset[i, y_Length - 1] = 2
        else:
            dataset[i, y_Length - 1] = 3

    for i in range(dl_Length):
        # date
        dataset[i, 0] = int(str(dataframe.iloc[i, 2]).replace("-", ""))
        # batter ID
        dataset[i, 1] = int(dataframe.iloc[i, 7])
        # pitcher ID
        dataset[i, 2] = int(dataframe.iloc[i, 8])

        # R is 0 and L is one
        if dataframe.iloc[i, 17] == "R":
            dataset[i, 3] = 0
        elif dataframe.iloc[i, 17] == "L":
            dataset[i, 3] = 1

        # R is 0 and L is one
        if dataframe.iloc[i, 18] == "R":
            dataset[i, 4] = 0
        elif dataframe.iloc[i, 18] == "L":
            dataset[i, 4] = 1

        # R is 0 and L is one
        if dataframe.iloc[i, 19] == "R":
            dataset[i, 5] = 0
        elif dataframe.iloc[i, 19] == "L":
            dataset[i, 5] = 1

        # Ball
        dataset[i, 6] = int(dataframe.iloc[i, 25])
        # Strike
        dataset[i, 7] = int(dataframe.iloc[i, 26])

        # On base
        # 3B
        if math.isnan(dataframe.iloc[i, 32]):

            dataset[i, 8] = 0
        else:
            dataset[i, 8] = 1

        # 2B
        if math.isnan(dataframe.iloc[i, 32]):
            dataset[i, 9] = 0
        else:
            dataset[i, 9] = 1
        # 1B
        if math.isnan(dataframe.iloc[i, 32]):
            dataset[i, 10] = 0
        else:
            dataset[i, 10] = 1

        # innings
        dataset[i, 11] = int(dataframe.iloc[i, 36])

        # outs
        dataset[i, 12] = int(dataframe.iloc[i, 35])

        # Catcher
        if math.isnan(dataframe.iloc[i, 61]):
            dataset[i, 13] = 0
        else:
            dataset[i, 13] = int(dataframe.iloc[i, 61])

        # Score
        # bat score
        dataset[i, 14] = int(dataframe.iloc[i, 82])
        # Field Score
        dataset[i, 15] = int(dataframe.iloc[i, 83])

        # Year/Career Pitches number
        dataset[i, 16] = dl_Length - 1 - i

        dataset[dl_Length - 1, 17] = 0
        dataset[dl_Length - 1, 18] = 0
        dataset[dl_Length - 1, 19] = 0
        dataset[dl_Length - 1, 20:y_Length - 4] = 0
        dataset[dl_Length - 1, y_Length - 3] = 999
        dataset[dl_Length - 1, y_Length - 2] = 1
        dataset[dl_Length - 2, y_Length - 2] = 1

    # game, out and inning pitch count
    for i in range(dl_Length - 2, -1, -1):
        # streak type
        dataset[i, y_Length - 3] = dataset[i + 1, y_Length - 1]
        # game
        if dataset[i, 0] != dataset[i + 1, 0]:
            dataset[i, 17] = 0
        else:
            dataset[i, 17] = dataset[i + 1, 17] + 1

        # inning
        if dataset[i, 11] != dataset[i + 1, 11]:
            dataset[i, 18] = 0
        else:
            dataset[i, 18] = dataset[i + 1, 18] + 1

        # out
        if dataset[i, 1] != dataset[i + 1, 1]:
            dataset[i, 19] = 0
        else:
            dataset[i, 19] = dataset[i + 1, 19] + 1

        # year/career pitch tracker
        dataset[i, 20:20 + num_Pitches] = dataset[i + 1, 20:20 + num_Pitches]
        dataset[i, 20 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 16] == 0:
            dataset[i, 20 + num_Pitches:20 + num_Pitches * 2] = 0
        else:
            dataset[i, 20 + num_Pitches:20 + num_Pitches * 2] = dataset[i, 20:20 + num_Pitches] / dataset[i, 16]

        # game pitch tracker
        if dataset[i + 1, 0] != dataset[i, 0]:
            dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] = 0
        else:
            dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] = dataset[i + 1,
                                                                    20 + num_Pitches * 2:20 + num_Pitches * 3]
            dataset[i, 20 + num_Pitches * 2 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 17] == 0:
            dataset[i, 20 + num_Pitches * 3:20 + num_Pitches * 4] = 0
        else:
            dataset[i, 20 + num_Pitches * 3:20 + num_Pitches * 4] = dataset[i,
                                                                    20 + num_Pitches * 2:20 + num_Pitches * 3] / \
                                                                    dataset[i, 17]

        # inning pitch tracker
        if dataset[i + 1, 11] != dataset[i, 11]:
            dataset[i, 20 + num_Pitches * 4:20 + num_Pitches * 5] = 0
        else:
            dataset[i, 20 + num_Pitches * 4:20 + num_Pitches * 5] = dataset[i + 1,
                                                                    20 + num_Pitches * 4:20 + num_Pitches * 5]
            dataset[i, 20 + num_Pitches * 4 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 18] == 0:
            dataset[i, 20 + num_Pitches * 5:20 + num_Pitches * 6] = 0
        else:
            dataset[i, 20 + num_Pitches * 5:20 + num_Pitches * 6] = dataset[i,
                                                                    20 + num_Pitches * 4:20 + num_Pitches * 5] / \
                                                                    dataset[i, 18]

        # out pitch tracker

        if dataset[i + 1, 1] != dataset[i, 1]:
            dataset[i, 20 + num_Pitches * 6:20 + num_Pitches * 7] = 0
        else:
            dataset[i, 20 + num_Pitches * 6:20 + num_Pitches * 7] = dataset[i + 1,
                                                                    20 + num_Pitches * 6:20 + num_Pitches * 7]
            dataset[i, 20 + num_Pitches * 6 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 19] == 0:
            dataset[i, 20 + num_Pitches * 7:20 + num_Pitches * 8] = 0
        else:
            dataset[i, 20 + num_Pitches * 7:20 + num_Pitches * 8] = dataset[i,
                                                                    20 + num_Pitches * 6:20 + num_Pitches * 7] / \
                                                                    dataset[i, 19]

            # streak length
    for i in range(dl_Length - 3, -1, -1):
        if dataset[i + 1, y_Length - 1] == dataset[i + 2, y_Length - 1]:
            dataset[i, y_Length - 2] = dataset[i + 1, y_Length - 2] + 1

    return dataset


def dataPipeBinary(path):
    # takes raw data from baseball stat scraper and turns it into a usable numpy array
    # seperates pitches into 2 pitches: fastball, offspeed
    dataframe = read_csv(path, header=0)
    pitches = dataframe.iloc[0:, 1]
    pitches_List = ["Fastball", "Breaking Ball", "Offspeed", "Other"]
    counter = 0

    dl_Length = len(pitches)
    num_Pitches = len(pitches_List)

    dataset = np.zeros([dl_Length, num_Pitches * 8 + 23])
    x_Length, y_Length = np.shape(dataset)
    for i in range(len(pitches)):
        if pitches[i] == "FF" or pitches[i] == "FT" or pitches[i] == "FS" or pitches[i] == "FC" or pitches[i] == "SI":
            # fastball
            dataset[i, y_Length - 1] = 0
        elif pitches[i] == "CU" or pitches[i] == "SL" or pitches[i] == "KC" or pitches[i] == "EP" or pitches[
            i] == "CH" or pitches[i] == "SF" or pitches[i] == "SC" or pitches[i] == "FO" or pitches[
            i] == "KN":  # breaking ball
            dataset[i, y_Length - 1] = 1
        else:
            dataset[i, y_Length - 1] = 2

    for i in range(dl_Length):
        # date
        dataset[i, 0] = int(str(dataframe.iloc[i, 2]).replace("-", ""))
        # batter ID
        dataset[i, 1] = int(dataframe.iloc[i, 7])
        # pitcher ID
        dataset[i, 2] = int(dataframe.iloc[i, 8])

        # R is 0 and L is one
        if dataframe.iloc[i, 17] == "R":
            dataset[i, 3] = 0
        elif dataframe.iloc[i, 17] == "L":
            dataset[i, 3] = 1

        # R is 0 and L is one
        if dataframe.iloc[i, 18] == "R":
            dataset[i, 4] = 0
        elif dataframe.iloc[i, 18] == "L":
            dataset[i, 4] = 1

        # R is 0 and L is one
        if dataframe.iloc[i, 19] == "R":
            dataset[i, 5] = 0
        elif dataframe.iloc[i, 19] == "L":
            dataset[i, 5] = 1

        # Ball
        dataset[i, 6] = int(dataframe.iloc[i, 25])
        # Strike
        dataset[i, 7] = int(dataframe.iloc[i, 26])

        # On base
        # 3B
        if math.isnan(dataframe.iloc[i, 32]):

            dataset[i, 8] = 0
        else:
            dataset[i, 8] = 1

        # 2B
        if math.isnan(dataframe.iloc[i, 32]):
            dataset[i, 9] = 0
        else:
            dataset[i, 9] = 1
        # 1B
        if math.isnan(dataframe.iloc[i, 32]):
            dataset[i, 10] = 0
        else:
            dataset[i, 10] = 1

        # innings
        dataset[i, 11] = int(dataframe.iloc[i, 36])

        # outs
        dataset[i, 12] = int(dataframe.iloc[i, 35])

        # Catcher
        if math.isnan(dataframe.iloc[i, 61]):
            dataset[i, 13] = 0
        else:
            dataset[i, 13] = int(dataframe.iloc[i, 61])

        # Score
        # bat score
        dataset[i, 14] = int(dataframe.iloc[i, 82])
        # Field Score
        dataset[i, 15] = int(dataframe.iloc[i, 83])

        # Year/Career Pitches number
        dataset[i, 16] = dl_Length - 1 - i

        dataset[dl_Length - 1, 17] = 0
        dataset[dl_Length - 1, 18] = 0
        dataset[dl_Length - 1, 19] = 0
        dataset[dl_Length - 1, 20:y_Length - 4] = 0
        dataset[dl_Length - 1, y_Length - 3] = -1
        dataset[dl_Length - 1, y_Length - 2] = 1
        dataset[dl_Length - 2, y_Length - 2] = 1

    # game, out and inning pitch count
    for i in range(dl_Length - 2, -1, -1):
        # streak type
        dataset[i, y_Length - 3] = dataset[i + 1, y_Length - 1]
        # game
        if dataset[i, 0] != dataset[i + 1, 0]:
            dataset[i, 17] = 0
        else:
            dataset[i, 17] = dataset[i + 1, 17] + 1

        # inning
        if dataset[i, 11] != dataset[i + 1, 11]:
            dataset[i, 18] = 0
        else:
            dataset[i, 18] = dataset[i + 1, 18] + 1

        # out
        if dataset[i, 1] != dataset[i + 1, 1]:
            dataset[i, 19] = 0
        else:
            dataset[i, 19] = dataset[i + 1, 19] + 1

        # year/career pitch tracker
        dataset[i, 20:20 + num_Pitches] = dataset[i + 1, 20:20 + num_Pitches]
        dataset[i, 20 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 16] == 0:
            dataset[i, 20 + num_Pitches:20 + num_Pitches * 2] = 0
        else:
            dataset[i, 20 + num_Pitches:20 + num_Pitches * 2] = int(
                100 * dataset[i, 20:20 + num_Pitches] / dataset[i, 16])

        # game pitch tracker
        if dataset[i + 1, 0] != dataset[i, 0]:
            dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] = 0
        else:
            dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] = dataset[i + 1,
                                                                    20 + num_Pitches * 2:20 + num_Pitches * 3]
            dataset[i, 20 + num_Pitches * 2 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 17] == 0:
            dataset[i, 20 + num_Pitches * 3:20 + num_Pitches * 4] = 0
        else:
            dataset[i, 20 + num_Pitches * 3:20 + num_Pitches * 4] = int(
                100 * dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] / \
                dataset[i, 17])

        # inning pitch tracker
        if dataset[i + 1, 11] != dataset[i, 11]:
            dataset[i, 20 + num_Pitches * 4:20 + num_Pitches * 5] = 0
        else:
            dataset[i, 20 + num_Pitches * 4:20 + num_Pitches * 5] = dataset[i + 1,
                                                                    20 + num_Pitches * 4:20 + num_Pitches * 5]
            dataset[i, 20 + num_Pitches * 4 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 18] == 0:
            dataset[i, 20 + num_Pitches * 5:20 + num_Pitches * 6] = 0
        else:
            dataset[i, 20 + num_Pitches * 5:20 + num_Pitches * 6] = int(100 * dataset[i,
                                                                              20 + num_Pitches * 4:20 + num_Pitches * 5] / \
                                                                        dataset[i, 18])

        # out pitch tracker

        if dataset[i + 1, 1] != dataset[i, 1]:
            dataset[i, 20 + num_Pitches * 6:20 + num_Pitches * 7] = 0
        else:
            dataset[i, 20 + num_Pitches * 6:20 + num_Pitches * 7] = dataset[i + 1,
                                                                    20 + num_Pitches * 6:20 + num_Pitches * 7]
            dataset[i, 20 + num_Pitches * 6 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 19] == 0:
            dataset[i, 20 + num_Pitches * 7:20 + num_Pitches * 8] = 0
        else:
            dataset[i, 20 + num_Pitches * 7:20 + num_Pitches * 8] = int(100 * dataset[i,
                                                                              20 + num_Pitches * 6:20 + num_Pitches * 7] / \
                                                                        dataset[i, 19])

            # streak length
    for i in range(dl_Length - 3, -1, -1):
        if dataset[i + 1, y_Length - 1] == dataset[i + 2, y_Length - 1]:
            dataset[i, y_Length - 2] = dataset[i + 1, y_Length - 2] + 1

    return dataset


def dataPipe(path):
    # takes raw data from baseball stat scraper and turns it into a usable numpy array
    # as many pitches as thrown
    dataframe = read_csv(path, header=0)
    pitches = dataframe.iloc[0:, 1]
    pitches_List = []
    counter = 0

    dl_Length = len(pitches)

    for i in range(len(pitches)):
        exists = False
        j = 0
        for j in range(len(pitches_List)):
            if pitches[i] == pitches_List[j]:
                exists = True
        if exists == False:
            pitches_List.append(pitches[i])
            counter += 1

    num_Pitches = len(pitches_List)

    dataset = np.zeros([dl_Length, num_Pitches * 8 + 23])
    x_Length, y_Length = np.shape(dataset)
    for i in range(len(pitches)):
        for j in range(len(pitches_List)):
            if pitches[i] == pitches_List[j]:
                dataset[i, y_Length - 1] = j
    for i in range(dl_Length):
        # date
        dataset[i, 0] = int(str(dataframe.iloc[i, 2]).replace("-", ""))
        # batter ID
        dataset[i, 1] = int(dataframe.iloc[i, 7])
        # pitcher ID
        dataset[i, 2] = int(dataframe.iloc[i, 8])

        # R is 0 and L is one
        if (dataframe.iloc[i, 17] == "R"):
            dataset[i, 3] = 0
        elif (dataframe.iloc[i, 17] == "L"):
            dataset[i, 3] = 1

        # R is 0 and L is one
        if (dataframe.iloc[i, 18] == "R"):
            dataset[i, 4] = 0
        elif (dataframe.iloc[i, 18] == "L"):
            dataset[i, 4] = 1

        # R is 0 and L is one
        if (dataframe.iloc[i, 19] == "R"):
            dataset[i, 5] = 0
        elif (dataframe.iloc[i, 19] == "L"):
            dataset[i, 5] = 1

        # Ball
        dataset[i, 6] = int(dataframe.iloc[i, 25])
        # Strike
        dataset[i, 7] = int(dataframe.iloc[i, 26])

        # On base
        # 3B
        if (math.isnan(dataframe.iloc[i, 32])):

            dataset[i, 8] = 0
        else:
            dataset[i, 8] = 1

        # 2B
        if (math.isnan(dataframe.iloc[i, 32])):
            dataset[i, 9] = 0
        else:
            dataset[i, 9] = 1
        # 1B
        if (math.isnan(dataframe.iloc[i, 32])):
            dataset[i, 10] = 0
        else:
            dataset[i, 10] = 1

        # innings
        dataset[i, 11] = int(dataframe.iloc[i, 36])

        # outs
        dataset[i, 12] = int(dataframe.iloc[i, 35])

        # Catcher
        if (math.isnan(dataframe.iloc[i, 61])):
            dataset[i, 13] = 0
        else:
            dataset[i, 13] = int(dataframe.iloc[i, 61])

        # Score
        # bat score
        dataset[i, 14] = int(dataframe.iloc[i, 82])
        # Field Score
        dataset[i, 15] = int(dataframe.iloc[i, 83])

        # Year/Career Pitches number
        dataset[i, 16] = dl_Length - 1 - i

        dataset[dl_Length - 1, 17] = 0
        dataset[dl_Length - 1, 18] = 0
        dataset[dl_Length - 1, 19] = 0
        dataset[dl_Length - 1, 20:y_Length - 4] = 0
        dataset[dl_Length - 1, y_Length - 3] = -1
        dataset[dl_Length - 1, y_Length - 2] = 1
        dataset[dl_Length - 2, y_Length - 2] = 1

    # game, out and inning pitch count
    for i in range(dl_Length - 2, -1, -1):
        # streak type
        dataset[i, y_Length - 3] = dataset[i + 1, y_Length - 1]
        # game
        if dataset[i, 0] != dataset[i + 1, 0]:
            dataset[i, 17] = 0
        else:
            dataset[i, 17] = dataset[i + 1, 17] + 1

        # inning
        if dataset[i, 11] != dataset[i + 1, 11]:
            dataset[i, 18] = 0
        else:
            dataset[i, 18] = dataset[i + 1, 18] + 1

        # out
        if dataset[i, 1] != dataset[i + 1, 1]:
            dataset[i, 19] = 0
        else:
            dataset[i, 19] = dataset[i + 1, 19] + 1

        # year/career pitch tracker
        dataset[i, 20:20 + num_Pitches] = dataset[i + 1, 20:20 + num_Pitches]
        dataset[i, 20 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 16] == 0:
            dataset[i, 20 + num_Pitches:20 + num_Pitches * 2] = 0
        else:
            dataset[i, 20 + num_Pitches:20 + num_Pitches * 2] = dataset[i, 20:20 + num_Pitches] / dataset[i, 16]

        # game pitch tracker
        if dataset[i + 1, 0] != dataset[i, 0]:
            dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] = 0
        else:
            dataset[i, 20 + num_Pitches * 2:20 + num_Pitches * 3] = dataset[i + 1,
                                                                    20 + num_Pitches * 2:20 + num_Pitches * 3]
            dataset[i, 20 + num_Pitches * 2 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 17] == 0:
            dataset[i, 20 + num_Pitches * 3:20 + num_Pitches * 4] = 0
        else:
            dataset[i, 20 + num_Pitches * 3:20 + num_Pitches * 4] = dataset[i,
                                                                    20 + num_Pitches * 2:20 + num_Pitches * 3] / \
                                                                    dataset[i, 17]

        # inning pitch tracker
        if dataset[i + 1, 11] != dataset[i, 11]:
            dataset[i, 20 + num_Pitches * 4:20 + num_Pitches * 5] = 0
        else:
            dataset[i, 20 + num_Pitches * 4:20 + num_Pitches * 5] = dataset[i + 1,
                                                                    20 + num_Pitches * 4:20 + num_Pitches * 5]
            dataset[i, 20 + num_Pitches * 4 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 18] == 0:
            dataset[i, 20 + num_Pitches * 5:20 + num_Pitches * 6] = 0
        else:
            dataset[i, 20 + num_Pitches * 5:20 + num_Pitches * 6] = dataset[i,
                                                                    20 + num_Pitches * 4:20 + num_Pitches * 5] / \
                                                                    dataset[i, 18]

        # out pitch tracker

        if dataset[i + 1, 1] != dataset[i, 1]:
            dataset[i, 20 + num_Pitches * 6:20 + num_Pitches * 7] = 0
        else:
            dataset[i, 20 + num_Pitches * 6:20 + num_Pitches * 7] = dataset[i + 1,
                                                                    20 + num_Pitches * 6:20 + num_Pitches * 7]
            dataset[i, 20 + num_Pitches * 6 + int(dataset[i + 1, y_Length - 1])] += 1

        # percentages
        if dataset[i, 19] == 0:
            dataset[i, 20 + num_Pitches * 7:20 + num_Pitches * 8] = 0
        else:
            dataset[i, 20 + num_Pitches * 7:20 + num_Pitches * 8] = dataset[i,
                                                                    20 + num_Pitches * 6:20 + num_Pitches * 7] / \
                                                                    dataset[i, 19]

            # streak length
    for i in range(dl_Length - 3, -1, -1):
        if dataset[i + 1, y_Length - 1] == dataset[i + 2, y_Length - 1]:
            dataset[i, y_Length - 2] = dataset[i + 1, y_Length - 2] + 1

    return dataset
