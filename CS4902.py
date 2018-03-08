import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn import preprocessing
from sklearn.model_selection import KFold

'''
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
Documentation on how the library and regressor works. Additional information on methods and parameters.
'''

raw_data = pd.read_csv('player_data2.csv')
player_ids = raw_data['player_id']
raw_data = raw_data.drop(['player_id'],axis=1)

#retain column names for later
col_names = list(raw_data)

def preprocess():

    #impute missing data using median
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(raw_data)

    #return as different object
    raw_data2 = imp.transform(raw_data)
    raw_data3 = preprocessing.scale(raw_data2)
    return raw_data3

def getData():
    dataFileNameTrain = 'Training.csv'  # Training data csv file path

    dataFileNameTest = 'Testing.csv'  # Testing data csv file path

    DataTrain = pd.read_csv(dataFileNameTrain)  # Uses the pandas library to format data for sklearn libraries

    DataTest = pd.read_csv(dataFileNameTest)

    y_train = DataTrain['ws']  # Insert column name of the target column and save the target column of training data

    X_train = DataTrain.drop(['ws'], axis=1)  # Drop the target column in training data

    y_test = DataTest['ws']  # Insert column name of the target column and save the target column of testing data

    X_test = DataTest.drop(['ws'], axis=1)  # Drop the target column in test data

    return X_train, y_train, X_test, y_test


def classify():
    X_train, y_train, X_test, y_test = getData()
    MLP = MLPRegressor(hidden_layer_sizes=(100))  # Change hidden_layer_sizes accordingly
    MLP.fit(X_train, y_train)
    predictions = MLP.predict(X_test)
                # This generates the predicted values. We can use k-fold cross validation to check accuracy, etc.
    return predictions, y_test


# Code to split data into training and testing datasets.

def kcv(dataSet, splitRatio):
    #randomly shuffle dataSet
    dataSet.sample(frac=1)

    player_ids = dataSet['player_id']
    dataSet = dataSet.drop(['player_id'],axis=1)

    #split according to ratio
    kf = KFold(n_splits = splitRatio)
    accuracies = []
    for train_ind, test_ind in kf.split(dataSet):
        trainSet = dataSet.iloc[train_ind]
        testSet = dataSet.iloc[test_ind]
        trainSet.to_csv('Training.csv')
        testSet.to_csv('Testing.csv')
        predictions, y_test = classify()
        accuracy = calc_rmse(predictions, y_test)
        accuracies.append(accuracy)
    av_acc = sum(accuracies)/float(len(accuracies))
    return av_acc	

def calc_rmse(predictions,y_test):
    accuracy = 0
    for i in range(len(predictions)):
        accuracy += (predictions[i]-y_test[i])**2
    accuracy = accuracy/len(predictions)
    accuracy = accuracy**(0.5)
    return accuracy

#preprocess on the raw_data
raw_data = pd.DataFrame(preprocess(),columns=col_names)
raw_data['player_id'] = player_ids

#initialize arrays to be used in plot
splitRatios = [2,3,4,5,6,7,8,9,10]
#trainingSizes = []
accuracies = []

#fill arrays
for i in range(len(splitRatios)):
    #trainingSizes.append(len(raw_data)*splitRatios[i])
    accuracy = kcv(raw_data,splitRatios[i])
    accuracies.append(accuracy)
    
plt.plot(splitRatios, accuracies)
plt.ylabel("RMSE")
plt.xlabel("Number of Splits")
plt.title("Learning Curve")
plt.savefig("Learning_Curve_4.png")
print(accuracies)
