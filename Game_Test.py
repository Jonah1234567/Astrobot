import numpy as np
from numpy import unique
from numpy import argmax
import math
from pandas import read_csv
from matplotlib import pyplot
from baseball_scraper import statcast
from baseball_scraper import playerid_lookup, statcast_pitcher,  pitching_stats, batting_stats_range

def cutoff(pred, real, score, percent):
    # takes predicting and real pitch values along with confidence values and will return accuracy on
    # a certain percent of guesses
    sampleNum, correct, cutoff = len(real), 1, 0.0001

    while (sampleNum / len(real)) > percent / 100:
        sampleNum, correct = 0, 0
        for i in range(len(pred)):
            if score[i, argmax(score[i])] > cutoff:
                sampleNum += 1
                if real[i] == pred[i]:
                    correct += 1
        cutoff += 0.0001

    if sampleNum and correct != 0:
        print(percent, "Percent Test: ", sampleNum, correct, correct / sampleNum * 100, "%, Cutoff Value:", cutoff)
        return correct / sampleNum * 100
    else:
        print("NaN")
        return -0.000001


def ones(pred, real, score):
    # sees how accurate confidence scores of 1 are
    sampleNum, correct = 0, 0

    for i in range(len(pred)):
        if score[i, argmax(score[i])] >= 1:
            sampleNum += 1
            if real[i] == pred[i]:
                correct += 1

    if sampleNum and correct != 0:
        print("Ones: ", sampleNum, correct, correct / sampleNum * 100, "%, Cutoff Value:")
        return correct / sampleNum * 100
    else:
        print("NaN")
        return -0.000001


def softmaxTest(yhat, y_test, probs):
    # runs cutoff and ones for various values
    sampleNum, correct = 0, 0
    print("Normal Test")
    for i in range(len(y_test)):
        sampleNum += 1
        if y_test[i] == yhat[i]:
            correct += 1

    if sampleNum and correct != 0:
        print(sampleNum, correct, correct / sampleNum * 100, "%")
    else:
        print("NaN")

    print("Cutoff Softmax Test")
    ninety = cutoff(yhat, y_test, probs, 90)
    seventy_five = cutoff(yhat, y_test, probs, 75)
    two_three = cutoff(yhat, y_test, probs, 66.6667)
    fifty = cutoff(yhat, y_test, probs, 50)
    one_three = cutoff(yhat, y_test, probs, 33.3333)
    twenty = cutoff(yhat, y_test, probs, 20)
    ten = cutoff(yhat, y_test, probs, 10)
    five = cutoff(yhat, y_test, probs, 5)
    three = cutoff(yhat, y_test, probs, 3)
    two = cutoff(yhat, y_test, probs, 2)
    oo = cutoff(yhat, y_test, probs, 1)

    one = ones(yhat, y_test, probs)

    true, false = 0, 0
    tCounter, fCounter = 0, 0
    for i in range(len(probs)):
        if (yhat[i] != y_test[i]):
            false += probs[i, argmax(probs[i])]
            fCounter += 1
            # print(probs[i,argmax(probs[i])],argmax(probs[i]), yhat[i],y_test[i])
        else:
            true += probs[i, argmax(probs[i])]
            tCounter += 1

    if true != 0 and tCounter != 0:
        print("True data average softmax value: ", true / tCounter)
    else:
        print("No Trues")
    if false != 0 and fCounter!=0:
        print("False data average softmax value: ", false / fCounter)
    else:
        print("No falses")
    return ninety, seventy_five, two_three, fifty, one_three, twenty, ten, five, one

def pitchTypeAnalysis(yhat, y_test):
    # looks at how accuracy differs based on predicted pitch and based on pitch thrown
    total, actual = 0,0
    fastball_pred, breaking_pred, changeup_pred = 0, 0, 0
    fastball_pred_sample, breaking_pred_sample, changeup_pred_sample = 0, 0, 0
    fastball_actual, breaking_actual, changeup_actual = 0, 0, 0
    fastball_actual_sample, breaking_actual_sample, changeup_actual_sample = 0, 0, 0

    for i in range(len(yhat)):
        total += 1
        if yhat[i] == y_test[i]:
            actual += 1

        if yhat[i] == 0:
            fastball_pred += 1
            if y_test[i] == 0:
                fastball_pred_sample += 1
        elif yhat[i] == 1:
            breaking_pred += 1
            if y_test[i] == 1:
                breaking_pred_sample += 1
        elif yhat[i] == 2:
            changeup_pred += 1
            if y_test[i] == 2:
                changeup_pred_sample += 1

        if y_test[i] == 0:
            fastball_actual += 1
            if yhat[i] == 0:
                fastball_actual_sample += 1
        elif y_test[i] == 1:
            breaking_actual += 1
            if yhat[i] == 1:
                breaking_actual_sample += 1
        elif y_test[i] == 2:
            breaking_actual += 1
            if yhat[i] == 2:
                breaking_actual_sample += 1

    print(actual/total*100, " Percent on all pitches")

    if fastball_pred != 0:
        print(fastball_pred_sample / fastball_pred * 100, " Percent on predicted fastballs")
    else:
        print("0 fastballs predicted")

    if fastball_actual != 0:
        print(fastball_actual_sample / fastball_actual * 100, " Percent on actual fastballs")
    else:
        print("0 fastballs thrown")

    if breaking_pred != 0:
        print(breaking_pred_sample / breaking_pred * 100, " Percent on predicted breaking balls")
    else:
        print("0 breaking balls predicted")

    if breaking_actual != 0:
        print(breaking_actual_sample / breaking_actual * 100, " Percent on actual breaking balls")
    else:
        print("0 breaking balls thrown")

    if changeup_pred != 0:
        print(changeup_pred_sample / changeup_pred * 100, " Percent on predicted change ups")
    else:
        print("0 change ups predicted")

    if changeup_actual != 0:
        print(changeup_actual_sample / changeup_actual * 100, " Percent on actual change ups")
    else:
        print("0 change ups thrown")

#Inning

#Pitch Num (Game, Inning, Out)







