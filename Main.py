import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from Utils.utils import game_train_NN, game_train_global_NN, individualMassTestSoftmax
from Testing.Game_Test import softmaxTest, cutoff, ones
from Testing.Feature_Testing import ablation
from Testing.Feature_Testing import linear_regression_report
from Data_Aquisition.Data_Aquisition import dataPipe3, scrape_pitch_data
from Models.Neural_Network import modelRunTest
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

CUDA_VISIBLE_DEVICES = 0

# linear_regression_report("2021-04-01", "2021-6-24")


# Function to create model, required for KerasClassifier
def create_model(n_features):
	model = Sequential()
	model.add(Dense(90, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(70, activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(4, activation='softmax'))
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
data = scrape_pitch_data("2021-06-01", "2021-06-23")
# create model
data_X, data_y = data[:, 1:-1], data[:, -1]
X, Y = data_X.astype('float'), data_y.astype('float')
n_features = data_X.shape[1]

# create model
model = KerasClassifier(build_fn=create_model(n_features), verbose=1)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))