from Models.Neural_Network import modelRunTest, modelRunTrain
from Models.Naive_Bayesian import naive_bayes
from Models.KNN import knn
from Models.SVM import support_vector_machine
from Models.Linear_Regression import linear_regression
import numpy as np
from Data_Aquisition.Data_Aquisition import scrape_pitch_data_player, scrape_pitch_data_player_binary, \
    scrape_pitch_data_player_Categorical
from Data_Aquisition.Data_Aquisition import scrape_pitch_data
from Testing.Game_Test import softmaxTest


def game_train_NN(start_date, end_date, first_name, last_name, e, bs, v, o):
    # training for local model of single player with a neural network

    data = scrape_pitch_data_player(start_date, end_date, first_name, last_name)

    split = 0
    for i in range(len(data) - 1):
        if data[i, 0] != data[i + 1, 0]:
            split = i
            break
    test_data = data[0:split + 1]
    train_data = data[split + 1:]

    acc, yhat, y_test, probs, model = modelRunTest(train_data, test_data, e, bs, v, o)
    return acc, yhat, y_test, probs, model


def game_train_global_NN(start_date, end_date, split, e, bs, v, o):
    # training for global model of with a neural network
    data = scrape_pitch_data(start_date, end_date)

    test_data = data[0:split + 1]
    train_data = data[split + 1:]

    acc, yhat, y_test, probs, model = modelRunTest(train_data, test_data, e, bs, v, o)
    return acc, yhat, y_test, probs, model


def game_train_NB(start_date, end_date, first_name, last_name):
    # training for local model of single player with a naive bayesian model, has some glitches right now
    data = scrape_pitch_data_player_Categorical(start_date, end_date, first_name, last_name)

    split = 0
    for i in range(len(data) - 1):
        if data[i, 0] != data[i + 1, 0]:
            split = i
            break
    test_data = data[0:split + 1]
    train_data = data[split + 1:]

    acc, yhat, y_test = naive_bayes(train_data, test_data)
    return acc, yhat, y_test


def game_train_KNN(start_date, end_date, first_name, last_name):
    # training for local model of single player with a k nearest neighbour model
    data = scrape_pitch_data_player_Categorical(start_date, end_date, first_name, last_name)

    split = 0
    for i in range(len(data) - 1):
        if data[i, 0] != data[i + 1, 0]:
            split = i
            break
    test_data = data[0:split + 1]
    train_data = data[split + 1:]

    acc, yhat, y_test, model = knn(train_data, test_data)
    return acc, yhat, y_test, model


def game_train_SVM(start_date, end_date, first_name, last_name, kernel):
    # training for local model of single player with a support vector machine, really slow to train currently
    data = scrape_pitch_data_player_Categorical(start_date, end_date, first_name, last_name)

    split = 0
    for i in range(len(data) - 1):
        if data[i, 0] != data[i + 1, 0]:
            split = i
            break
    test_data = data[0:split + 1]
    train_data = data[split + 1:]

    acc, yhat, y_test, model = support_vector_machine(train_data[0:250], test_data, kernel)
    return acc, yhat, y_test, model


def game_train_LR(start_date, end_date, first_name, last_name):
    # training for local model of single player with a linear regression
    data = scrape_pitch_data_player_Categorical(start_date, end_date, first_name, last_name)

    split = 0
    for i in range(len(data) - 1):
        if data[i, 0] != data[i + 1, 0]:
            split = i
            break
    test_data = data[0:split + 1]
    train_data = data[split + 1:]

    acc, yhat, y_test, model = linear_regression(train_data, test_data)
    return acc, yhat, y_test, model


def game_train_binary(start_date, end_date, first_name, last_name, e, bs, v, o):
    # training for local model of single player with a neural network for binary values (fastball offspeed)
    data = scrape_pitch_data_player_binary(start_date, end_date, first_name, last_name)

    split = 0
    for i in range(len(data) - 1):
        if data[i, 0] != data[i + 1, 0]:
            split = i
            break
    test_data = data[0:split + 1]
    train_data = data[split + 1:]

    acc, yhat, y_test, probs, model = modelRunTest(train_data, test_data, e, bs, v, o)
    return acc, yhat, y_test, probs, model


def individualMassTest(start, end, e, bs, v, op):
    # testing of top ~50 pitchers with local neural network model
    name = [
        'Ryan', 'Yarbrough', 'Logan', 'Webb', 'Freddy', 'Peralta', 'David', 'Price', 'Jacob', 'deGrom', 'Shane',
        'Bieber',
        'Gerrit', 'Cole', 'Yu', 'Darvish', 'Lucas', 'Giolito', 'Aaron', 'Nola', 'Walker', 'Buehler', 'Max', 'Scherzer',
        'Clayton', 'Kershaw', 'Kenta', 'Maeda', 'Brandon', 'Woodruff', 'Blake', 'Snell', 'Lance', 'Lynn', 'Tyler',
        'Glasnow',
        'Zack', 'Wheeler', 'Hyun Jin', 'Ryu', 'Max', 'Fried', 'Zach', 'Plesac', 'Corbin', 'Burnes', 'Stephen',
        'Strasburg',
        'Patrick', 'Corbin', 'Chris', 'Paddack', 'Zack', 'Greinke', 'Jose', 'Berrios', 'Sandy', 'Alcantara', 'Sixto',
        'Sanchez', 'Jesus', 'Luzardo', 'Pablo', 'Lopez', 'Julio', 'Urias', 'Aaron', 'Civale', 'Corey', 'Kluber',
        'Jameson', 'Taillon', 'Ian', 'Anderson', 'Frankie', 'Montas', 'Joe', 'Musgrove', 'Dylan', 'Bundy', 'Tyler',
        'Mahle',
        'Jose', 'Urquidy', 'John', 'Means', 'Kevin', 'Gausman', 'Marcus', 'Stroman', 'Shohei', 'Ohtani', 'James',
        'Paxton',
        'Triston', 'McKenzie', 'Domingo', 'German', 'Dallas', 'Keuchel', 'Marco', 'Gonzales', 'Zach', 'Davies',
        'Drew', 'Smyly', 'Mike', 'Minor'
    ]

    total, i = 0, 0
    k = np.zeros((int(len(name) / 2)))
    naive = np.zeros((int(len(name) / 2)))
    for i in range(0, len(name), 2):
        print(name[i], name[i + 1])
        a, b, c, d, e = game_train_NN(start, end, name[i], name[i + 1], e, bs, v, op)
        print(a, " model")
        naiveC, naiveT = 0, 0
        for l in range(len(c)):
            if c[l] == 0:
                naiveC += 1
            naiveT += 1
        print(naiveC / naiveT, "naive")
        total += a
        k[int(i / 2)] = a
        naive[int(i / 2)] = naiveC / naiveT

    print(total)
    print(2 * total / len(name))
    print("Total: ", np.sum(k[:]) / (len(name) / 2))
    print("naive: ", np.sum(naive[:]) / (len(name) / 2))


def individualMassTestSoftmax(start, end, e, bs, v, op):
    # testing of top ~50 pitchers with local neural network model, softmax measurements built in
    name = [
        'Ryan', 'Yarbrough', 'Logan', 'Webb', 'Freddy', 'Peralta', 'David', 'Price', 'Jacob', 'deGrom', 'Shane',
        'Bieber',
        'Gerrit', 'Cole', 'Yu', 'Darvish', 'Lucas', 'Giolito', 'Aaron', 'Nola', 'Walker', 'Buehler', 'Max', 'Scherzer',
        'Clayton', 'Kershaw', 'Kenta', 'Maeda', 'Brandon', 'Woodruff', 'Blake', 'Snell', 'Lance', 'Lynn', 'Tyler',
        'Glasnow',
        'Zack', 'Wheeler', 'Hyun Jin', 'Ryu', 'Max', 'Fried', 'Zach', 'Plesac', 'Corbin', 'Burnes', 'Stephen',
        'Strasburg',
        'Patrick', 'Corbin', 'Chris', 'Paddack', 'Zack', 'Greinke', 'Jose', 'Berrios', 'Sandy', 'Alcantara', 'Sixto',
        'Sanchez', 'Jesus', 'Luzardo', 'Pablo', 'Lopez', 'Julio', 'Urias', 'Aaron', 'Civale', 'Corey', 'Kluber',
        'Jameson', 'Taillon', 'Ian', 'Anderson', 'Frankie', 'Montas', 'Joe', 'Musgrove', 'Dylan', 'Bundy', 'Tyler',
        'Mahle',
        'Jose', 'Urquidy', 'John', 'Means', 'Kevin', 'Gausman', 'Marcus', 'Stroman', 'Shohei', 'Ohtani', 'James',
        'Paxton',
        'Triston', 'McKenzie', 'Domingo', 'German', 'Dallas', 'Keuchel', 'Marco', 'Gonzales', 'Zach', 'Davies',
        'Drew', 'Smyly', 'Mike', 'Minor'
    ]

    total, i = 0, 0
    k = np.zeros((int(len(name) / 2), 10))
    naive = np.zeros((int(len(name) / 2)))
    for i in range(0, len(name), 2):
        print(name[i], name[i + 1])
        a, b, c, d, model = game_train_NN(start, end, name[i], name[i + 1], e, bs, v, op)
        k[int(i / 2), 1:] = softmaxTest(b, c, d)

        print(a, " model")
        naiveC, naiveT = 0, 0
        for l in range(len(c)):
            if c[l] == 0:
                naiveC += 1
            naiveT += 1
        print(naiveC / naiveT, "naive")
        total += a
        k[int(i / 2), 0] = a
        naive[int(i / 2)] = naiveC / naiveT

    print(total)
    print(2 * total / len(name))
    print("Total: ", np.sum(k[:, 0]) / (len(name) / 2))
    print("90: ", np.sum(k[:, 1]) / (len(name) / 2))
    print("75: ", np.sum(k[:, 2]) / (len(name) / 2))
    print("2/3: ", np.sum(k[:, 3]) / (len(name) / 2))
    print("50: ", np.sum(k[:, 4]) / (len(name) / 2))
    print("1/3: ", np.sum(k[:, 5]) / (len(name) / 2))
    print("20: ", np.sum(k[:, 6]) / (len(name) / 2))
    print("10: ", np.sum(k[:, 7]) / (len(name) / 2))
    print("5: ", np.sum(k[:, 8]) / (len(name) / 2))
    print("naive: ", np.sum(naive[:]) / (len(name) / 2))
    print("1: ", np.sum(k[:, 9]) / (len(name) / 2))


def individualMassTestBinary(start, end, e, bs, v, op):
    # testing of top ~50 pitchers with local neural network model for binary values
    name = [
        'Ryan', 'Yarbrough', 'Logan', 'Webb', 'Freddy', 'Peralta', 'David', 'Price', 'Jacob', 'deGrom', 'Shane',
        'Bieber',
        'Gerrit', 'Cole', 'Yu', 'Darvish', 'Lucas', 'Giolito', 'Aaron', 'Nola', 'Walker', 'Buehler', 'Max', 'Scherzer',
        'Clayton', 'Kershaw', 'Kenta', 'Maeda', 'Brandon', 'Woodruff', 'Blake', 'Snell', 'Lance', 'Lynn', 'Tyler',
        'Glasnow',
        'Zack', 'Wheeler', 'Hyun Jin', 'Ryu', 'Max', 'Fried', 'Zach', 'Plesac', 'Corbin', 'Burnes', 'Stephen',
        'Strasburg',
        'Patrick', 'Corbin', 'Chris', 'Paddack', 'Zack', 'Greinke', 'Jose', 'Berrios', 'Sandy', 'Alcantara', 'Sixto',
        'Sanchez', 'Jesus', 'Luzardo', 'Pablo', 'Lopez', 'Julio', 'Urias', 'Aaron', 'Civale', 'Corey', 'Kluber',
        'Jameson', 'Taillon', 'Ian', 'Anderson', 'Frankie', 'Montas', 'Joe', 'Musgrove', 'Dylan', 'Bundy', 'Tyler',
        'Mahle',
        'Jose', 'Urquidy', 'John', 'Means', 'Kevin', 'Gausman', 'Marcus', 'Stroman', 'Shohei', 'Ohtani', 'James',
        'Paxton',
        'Triston', 'McKenzie', 'Domingo', 'German', 'Dallas', 'Keuchel', 'Marco', 'Gonzales', 'Zach', 'Davies',
        'Drew', 'Smyly', 'Mike', 'Minor'
    ]

    total, i = 0, 0
    k = np.zeros((int(len(name) / 2)))
    naive = np.zeros((int(len(name) / 2)))
    for i in range(0, len(name), 2):
        print(name[i], name[i + 1])
        a, b, c, d, e = game_train_binary(start, end, name[i], name[i + 1], e, bs, v, op)
        print(a, " model")
        naiveC, naiveT = 0, 0
        for l in range(len(c)):
            if c[l] == 0:
                naiveC += 1
            naiveT += 1
        print(naiveC / naiveT, "naive")
        total += a
        k[int(i / 2)] = a
        naive[int(i / 2)] = naiveC / naiveT

    print(total)
    print(2 * total / len(name))
    print("Total: ", np.sum(k[:]) / (len(name) / 2))
    print("naive: ", np.sum(naive[:]) / (len(name) / 2))
    plt.hist(k[:], bins=10)
    plt.show()


def test(data):
    split = 0
    for i in range(len(data) - 1):
        if data[i, 0] != data[i + 1, 0]:
            split = i
            break
    test_data = data[0:split + 1]
    train_data = data[split + 1:]

    acc, yhat, y_test, probs, model = modelRunTest(train_data, test_data, 20, 32, 0)
    return acc, yhat, y_test, probs, model
