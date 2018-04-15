import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn import preprocessing
from sklearn.model_selection import KFold
import numpy as np

'''
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
Documentation on how the library and regressor works. Additional information on methods and parameters.
'''

raw_data = pd.read_csv('player_data_career_without_2017.csv')
# topRow = raw_data[0]
# drop unnecessary columns
raw_data = raw_data.drop(['player_id'], axis=1)
#raw_data = raw_data.drop(['Age5'], axis=1)
#raw_data = raw_data.drop(['College5'], axis=1)
#raw_data = raw_data.drop(['name'], axis=1)
# retain column names for later
col_names = list(raw_data)
print(col_names)

def preprocess():
    # change categorical data to numerical
    le_name = LabelEncoder()
    raw_data['name'] = le_name.fit_transform(raw_data['name'].astype(str))

    #le_s1 = LabelEncoder()
    #raw_data['Season1'] = le_s1.fit_transform(raw_data['Season1'].astype(str))

    #le_s2 = LabelEncoder()
    #raw_data['Season2'] = le_s2.fit_transform(raw_data['Season2'].astype(str))

    #le_s3 = LabelEncoder()
    #raw_data['Season3'] = le_s3.fit_transform(raw_data['Season3'].astype(str))

    #le_s4 = LabelEncoder()
    #raw_data['Season4'] = le_s4.fit_transform(raw_data['Season4'].astype(str))

    #le_s5 = LabelEncoder()
    #raw_data['Season5'] = le_s5.fit_transform(raw_data['Season5'].astype(str))

    #le_c4 = LabelEncoder()
    #raw_data['College4'] = le_c4.fit_transform(raw_data['College4'].astype(str))

    #le_c3 = LabelEncoder()
    #raw_data['College3'] = le_c3.fit_transform(raw_data['College3'].astype(str))

    #le_c2 = LabelEncoder()
    #raw_data['College2'] = le_c2.fit_transform(raw_data['College2'].astype(str))

    #le_c1 = LabelEncoder()
    #raw_data['College1'] = le_c1.fit_transform(raw_data['College1'].astype(str))

    # impute missing data using median
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(raw_data)

    # return as different object
    raw_data2 = imp.transform(raw_data)
    raw_data3 = preprocessing.scale(raw_data2)
    # fileOut = open('normalizedData.csv', 'w')
    # writer = csv.writer(fileOut, delimiter=',')
    # for row in raw_data3:
    #    writer.writerow(row)
    #    raw_data3.insert(0, topRow)
    #    print(raw_data3[0])
    #    print(raw_data3[1])
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


def classify(flag, data):
    X_train, y_train, X_test, y_test = getData()
    print(flag, len(X_train), len(y_train), len(X_test), len(y_test))
    MLP = MLPRegressor(
        hidden_layer_sizes=(300, 300))  # Change hidden_layer_sizes accordingly
    MLP.fit(X_train, y_train)
    if flag is True:
        predictions = MLP.predict(data)
        return predictions
    else:
        predictions = MLP.predict(X_test)
        # This generates the predicted values. We can use k-fold cross validation to check accuracy, etc.
        return predictions, y_test


# Code to split data into training and testing datasets.

def kcv(dataSet, splitRatio):
    # randomly shuffle dataSet
    dataSet.sample(frac=1)

    # split according to ratio
    kf = KFold(n_splits=splitRatio)
    accuracies = []
    i = 0
    for train_ind, test_ind in kf.split(dataSet):
        trainSet = dataSet.iloc[train_ind]
        testSet = dataSet.iloc[test_ind]
        trainSet.to_csv('Training.csv')
        testSet.to_csv('Testing.csv')
        predictions, y_test = classify(False, None)
        accuracy = calc_rmse(predictions, y_test)
        accuracies.append(accuracy)
    av_acc = sum(accuracies) / float(len(accuracies))
    return av_acc


def calc_rmse(predictions, y_test):
    accuracy = 0
    for i in range(len(predictions)):
        accuracy += (predictions[i] - y_test[i]) ** 2
    accuracy = accuracy / len(predictions)
    accuracy = accuracy ** (0.5)
    return accuracy


# preprocess on the raw_data
raw_data = pd.DataFrame(preprocess(), columns=col_names)

# initialize arrays to be used in plot
splitRatios = [9]
# trainingSizes = []
accuracies = []

# fill arrays
for i in range(len(splitRatios)):
    # trainingSizes.append(len(raw_data)*splitRatios[i])
    accuracy = kcv(raw_data, splitRatios[i])
    accuracies.append(accuracy)

##2017
print('Done training!!!!')
print()
print('DRAFT CLASS 2010')
data_2017 = pd.read_csv('draft_class_2017_career.csv')
# topRow = raw_data[0]
# drop unnecessary columns
data_2017 = data_2017.drop(['player_id'], axis=1)
#data_2017 = data_2017.drop(['Age5'], axis=1)
#data_2017 = data_2017.drop(['College5'], axis=1)
#data_2017 = data_2017.drop(['ws'], axis=1)
#data_2017 = data_2017.drop(['X2'], axis=1)
#data_2017 = data_2017.drop(['X1'], axis=1)
#data_2017 = data_2017.drop(['X'], axis=1)
data_2017 = data_2017.drop(['U1'], axis=1)

data_2017_names = data_2017['name']
col_names2 = list(data_2017)
print(col_names2)

le_name = LabelEncoder()
data_2017['name'] = le_name.fit_transform(data_2017['name'].astype(str))

#le_s1 = LabelEncoder()
#data_2017['Season1'] = le_s1.fit_transform(data_2017['Season1'].astype(str))

#le_s2 = LabelEncoder()
#data_2017['Season2'] = le_s2.fit_transform(data_2017['Season2'].astype(str))

#le_s3 = LabelEncoder()
#data_2017['Season3'] = le_s3.fit_transform(data_2017['Season3'].astype(str))

#le_s4 = LabelEncoder()
#data_2017['Season4'] = le_s4.fit_transform(data_2017['Season4'].astype(str))

#le_s5 = LabelEncoder()
#data_2017['Season5'] = le_s5.fit_transform(data_2017['Season5'].astype(str))

#le_c4 = LabelEncoder()
#data_2017['College4'] = le_c4.fit_transform(data_2017['College4'].astype(str))

#le_c3 = LabelEncoder()
#data_2017['College3'] = le_c3.fit_transform(data_2017['College3'].astype(str))

#le_c2 = LabelEncoder()
#data_2017['College2'] = le_c2.fit_transform(data_2017['College2'].astype(str))

#le_c1 = LabelEncoder()
#data_2017['College1'] = le_c1.fit_transform(data_2017['College1'].astype(str))

# impute missing data using median
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(data_2017)

# return as different object
raw_data2 = imp.transform(data_2017)
#print(list(raw_data2))
data_2017 = preprocessing.scale(raw_data2)

preds = classify(True, data_2017)
print(len(preds))
predictions = preds.tolist()
print(predictions)

print('type', type(predictions))
player_score = {}
for j in range(len(predictions)):
    player_score[predictions[j]] = data_2017_names[j]
predictions.sort()
scores = predictions
print('PREDICTIONS!!!!')
print('-------------')
# rankings = open('Rankings.csv', 'w')
# writer = csv.writer(rankings)
# i = 1
# for p in predictions:
#    writer.writerow(str(p))
#    i+= 1
print(type(scores))
for s in scores:
    print(player_score[s])
for s in scores:
    print(s)
print('DONE!', len(scores))
'''

plt.plot(splitRatios, accuracies)
plt.ylabel("RMSE")
plt.xlabel("Number of Splits")
plt.title("Learning Curve")
plt.savefig("Learning_Curve_4.png")
for a in accuracies:
    print("{0:.3f}".format(a))
'''
