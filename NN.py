import pandas as pd
import csv
import random
from sklearn.neural_network import MLPRegressor

'''
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
Documentation on how the library and regressor works. Additional information on methods and parameters.
'''

def getData():
    dataFileNameTrain = 'Training.csv'  # Training data csv file path

    dataFileNameTest = 'Testing.csv'  # Testing data csv file path

    DataTrain = pd.read_csv(dataFileNameTrain)  # Uses the pandas library to format data for sklearn libraries

    DataTest = pd.read_csv(dataFileNameTest)

    y_train = DataTrain['']  # Insert column name of the target column and save the target column of training data

    X_train = DataTrain.drop([''], axis=1)  # Drop the target column in training data

    y_test = DataTest['']  # Insert column name of the target column and save the target column of testing data

    X_test = DataTest.drop([''], axis=1)  # Drop the target column in test data

    return X_train, y_train, X_test, y_test


def classify():
    X_train, y_train, X_test, y_test = getData()
    MLP = MLPRegressor(hidden_layer_sizes=(30, 30, 30))  # Change hidden_layer_sizes accordingly
    MLP.fit(X_train, y_train)
    predictions = MLP.predict(X_test)
                # This generates the predicted values. We can use k-fold cross validation to check accuracy, etc.
    return predictions


# Code to split data into training and testing datasets.

def splitDataset(dataSet, splitRatio):
    random.shuffle(dataSet)
    trainSize = int(len(dataSet) * splitRatio)
    trainSet = dataSet[:trainSize]
    testSet = dataSet[trainSize:]
    return trainSet, testSet
