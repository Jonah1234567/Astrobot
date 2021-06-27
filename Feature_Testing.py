from Data_Aquisition.Data_Aquisition import scrape_pitch_data
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import LinearRegression
from Models.Neural_Network import modelRunTest

def linear_regression_report(sdate, edate):
    data = scrape_pitch_data(sdate, edate)
    data_X, data_y = data[:, 1:-1], data[:, -1]
    data_X, data_y = data_X.astype('float'), data_y.astype('float')
    data_X = preprocessing.scale(data_X)

    x, y = np.shape(data_X)

    for i in range(y):
        dx = data_X[i]
        reg = LinearRegression().fit(dx, data_y)
        print("Column ", i, " R squared score of: ", reg.score(dx, data_y))

def ablation(start_date, end_date, split,  e, bs, v, o):
    data = scrape_pitch_data(start_date, end_date)
    x, y = np.shape(data)
    scores = np.zeros(y)

    for i in range(y):
        test_data = data[0:split + 1]
        train_data = data[split + 1:]
        train_data = np.delete(train_data, 0, i)
        acc, yhat, y_test, probs, model = modelRunTest(train_data, test_data, e, bs, v, o)
        scores[i] = acc
        print("Column No. ", i, " Acc: ", acc)

    for i in range(y):
        print("Column No. ", i, " Acc: ", scores[i])