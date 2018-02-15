import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, Imputer

'''
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
Documentation on how the library and regressor works. Additional information on methods and parameters.
'''

raw_data = pd.read_csv('player_data.csv')
raw_data = raw_data.drop(['player_id'], axis=1)
raw_data = raw_data.drop(['Age5'], axis=1)
raw_data = raw_data.drop(['College5'], axis=1)
col_names = list(raw_data)

def preprocess():
    le_name = LabelEncoder()
    raw_data['name'] = le_name.fit_transform(raw_data['name'].astype(str))

    le_s1 = LabelEncoder()
    raw_data['Season1'] = le_s1.fit_transform(raw_data['Season1'].astype(str))

    le_s2 = LabelEncoder()
    raw_data['Season2'] = le_s2.fit_transform(raw_data['Season2'].astype(str))

    le_s3 = LabelEncoder()
    raw_data['Season3'] = le_s3.fit_transform(raw_data['Season3'].astype(str))

    le_s4 = LabelEncoder()
    raw_data['Season4'] = le_s4.fit_transform(raw_data['Season4'].astype(str))

    le_s5 = LabelEncoder()
    raw_data['Season5'] = le_s5.fit_transform(raw_data['Season5'].astype(str)) 
    
    le_c4 = LabelEncoder()
    raw_data['College4'] = le_c4.fit_transform(raw_data['College4'].astype(str))

    le_c3 = LabelEncoder()
    raw_data['College3'] = le_c3.fit_transform(raw_data['College3'].astype(str))

    le_c2 = LabelEncoder()
    raw_data['College2'] = le_c2.fit_transform(raw_data['College2'].astype(str))

    le_c1 = LabelEncoder()
    raw_data['College1'] = le_c1.fit_transform(raw_data['College1'].astype(str))

    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(raw_data)
    raw_data2 = imp.transform(raw_data)
    return raw_data2

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
    MLP = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100))  # Change hidden_layer_sizes accordingly
    MLP.fit(X_train, y_train)
    predictions = MLP.predict(X_test)
                # This generates the predicted values. We can use k-fold cross validation to check accuracy, etc.
    return predictions, y_test


# Code to split data into training and testing datasets.

def splitDataset(dataSet, splitRatio):
    #random.shuffle(dataSet)
    dataSet.sample(frac=1)
    trainSize = int(len(dataSet) * splitRatio)
    trainSet = dataSet[:trainSize]
    testSet = dataSet[trainSize:]
    return trainSet, testSet

def saveSplitDataset(splitRatio):
    #call splitDataset and 
    trainSet,testSet = splitDataset(raw_data,splitRatio)
    trainSet.to_csv('Training.csv')
    testSet.to_csv('Testing.csv')

splitRatios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
def main(splitRatio):
    saveSplitDataset(splitRatio)
    predictions, y_test = classify()
    accuracy = 0
    for i in range(len(predictions)):
        accuracy += predictions[i] - y_test[i]
    return accuracy

raw_data = pd.DataFrame(preprocess(),columns=col_names)
trainingSizes = []
accuracies = []
for i in range(len(splitRatios)):
    trainingSizes.append(len(raw_data)*splitRatios[i])
    accuracy = main(splitRatios[i])
    accuracies.append(accuracy)
    
plt.plot(splitRatios, accuracies)
plt.ylabel("Accuracy")
plt.xlabel("Training Set Size")
plt.title("Learning Curve")
plt.savefig("Learning_Curve_4.png")
print(accuracies)
