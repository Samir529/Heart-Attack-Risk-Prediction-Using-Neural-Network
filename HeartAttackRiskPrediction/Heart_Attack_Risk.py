

# Importing

import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix



# Read The Dataset
def get_dataset(filename):
    return pd.read_csv(filename)



# Printing the dataset
def print_data(data,n):
    print(data.head(n))




# Cleaning The Dataset

### Detecting The None/Missing Values




def count(a):
    count1 = 0
    for i in a:
        if (i >= -np.inf and i <= np.inf):
            count1 = count1 + 1

    return count1

# Removing The Duplicate Values

def data_cleaning(data):
    # Total rows
    total_rows = len(data.index)
    print("Total Rows of The Dataset", total_rows)

    # Removing Duplicate Rows
    data.drop_duplicates(keep=False, inplace=True)

    # Total rows after removing duplicates
    total_rows_after_remove_duplicates = len(data.index)

    print("Total Rows of The Dataset After Removing Duplicates", total_rows_after_remove_duplicates)
    print("There are {} duplicates in the dataset".format(total_rows - total_rows_after_remove_duplicates))
    return data

def data_is_anyMissingData(data):
    for j in data.columns:
        c = count(data[j])
        if (c == len(data.index)):
            print("There is no missing value for column", j)
        else:
            print("None/Missing values detected for column", j)

# There is no missing value in the whole dataset



# Setting up the features with the columns of dataset except 'output'

def get_features(data):
    features = data.columns
    features = [i for i in features if i != 'output']
    # display columns of the feature
    print(features)
    return features



# Splitting the dataset into train and test dataset
def data_split_test_train(data):
    train, test = train_test_split(data, test_size=0.25)  # Train dataset is 75% of actual dataset
    print("Length Of Actual Heart Dataset: ", len(data))
    print("Length Of Train Dataset: ", len(train))
    print("Length Of Test Dataset: ", len(test))
    return train,test



# Setting Up The Train and Test Dataset Into x And y

def set_XY(train,test,features):
    x_train = train[features]
    y_train = train["output"]

    x_test = test[features]
    y_test = test["output"]

    return x_train,x_test,y_train,y_test


### Creating the neural network model

def NNModel():
    mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=900, activation='relu')
    return mlp


### Fitting the train and test dataset in the model

def fitModel(x_train,y_train,mlp):
    mlp = mlp.fit(x_train, y_train)
    return mlp



### Prediction according to neural network algorithm

def get_prediction(mlp,x_test):
    y_pred = mlp.predict(x_test)
    print(y_pred)
    return y_pred

### Determining the Accuracy for Neural Network

def model_accuracy(y_test, y_pred):
    score_NN = accuracy_score(y_test, y_pred) * 100
    print("Accuracy using Neural Network: ", round(score_NN, 4), "%")
    return score_NN

### Confusion Matrix For Neural Network Algorithm

def get_confusion_matrix(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix for Neural Network: ")
    print(conf_mat)
    return conf_mat

def users_test(mlp,user_test):
    user_pred = mlp.predict(user_test)
    if user_pred[0]==1:
        print("There is Heart Risk. Consult With Doctor ASAP")
    elif user_pred[0]==0:
        print("Good News,You have no Heart Attack Risk. But Keep in touch with your Doctor")
    # print("\n\nUSer Prediction:\n", user_pred)



# Plottings

### Splitting of train and test dataset


def test_train_ration_bar_chart(train,test):
    y = [len(train), len(test)]
    x = [0, 1]

    tick_label = ['Train Dataset', 'Test Dataset']

    plt.bar(x, y, tick_label=tick_label, width=0.6, color=['red', 'blue'])

    plt.title("Splitting The Dataset")
    plt.ylabel('Count Of The Dataset')
    plt.show()



### Pie Chart According to Exercise

def exercise_pie_chart(data):
    do_exercise = data.loc[(data['exng'] == 1) & (data['output'] == 1)]
    no_exercise = data.loc[(data['exng'] == 0) & (data['output'] == 1)]

    percentage_do_exercise = (do_exercise['output'].sum() / len(data.index)) * 100
    percentage_no_exercise = (no_exercise['output'].sum() / len(data.index)) * 100

    explode = (0.05, 0.05)
    label = 'Heart Attack Risk With Exercise', 'Heart Attack Risk Without Exercise'
    perc = [percentage_do_exercise, percentage_no_exercise]

    plt.pie(perc, explode=explode, labels=label, autopct='%1.2f%%', shadow=True)
    plt.axis('equal')
    plt.title('Percentage of Heart Attack Risk: Who Performs Exercise vs Who Does not Perform Exercise\n')
    plt.show()



### Heart Attack Number of Different Age Range in Dataset
def age_range_heartAttack_BarChart(data):
    age_ten_thirty = data.loc[(data['age'] >= 10) & (data['age'] <= 30) & (data['output'] == 1)]
    age_thirtyone_fifty = data.loc[(data['age'] >= 31) & (data['age'] <= 50) & (data['output'] == 1)]
    age_fiftyone_seventy = data.loc[(data['age'] >= 51) & (data['age'] <= 70) & (data['output'] == 1)]
    age_fiftyone_ninty = data.loc[(data['age'] >= 71) & (data['age'] <= 90) & (data['output'] == 1)]

    y = [len(age_ten_thirty), len(age_thirtyone_fifty), len(age_fiftyone_seventy), len(age_fiftyone_ninty)]
    x = [0, 1, 2, 3]

    tick_label = ['10 - 30', '31 - 50', '51 - 70', '71 - 90']

    plt.bar(x, y, tick_label=tick_label, width=0.6, color=['Green', 'purple', 'red', 'blue'])

    plt.ylabel("Number Of People Who Got Heart Attack")
    plt.xlabel('Age')
    plt.title('Age Wise Heart Attack Prediction')
    plt.show()


if __name__ == '__main__':

    user_test = [[23, 1, 3, 120, 200, 1, 140, 1, 0, 2.3, 0, 0, 1]]
    NumberOfParam = len(sys.argv)

    for i in range(1, NumberOfParam):
        if sys.argv[i].replace(' ', '') == "-ui":
            user_test = (sys.argv[i + 1]).strip('[]')

            user_test = list(map(float, user_test.split(",")))

            user_test = [user_test]


    filename="heart.csv"

    data = get_dataset(filename)
    print()
    print_data(data,11)
    print()
    data_is_anyMissingData(data)
    print()
    data=data_cleaning(data)
    print()

    features = get_features(data)
    print()
    train, test = data_split_test_train(data)
    print()
    x_train, x_test, y_train, y_test = set_XY(train, test, features)
    print()
    mlp = NNModel()
    mlp = fitModel(x_train, y_train, mlp)
    print()
    y_pred = get_prediction(mlp, x_test)
    print()
    conf_mat = get_confusion_matrix(y_test, y_pred)
    print()
    score_NN = model_accuracy(y_test, y_pred)
    print()

    users_test(mlp, user_test)

    # plottings
    test_train_ration_bar_chart(train, test)

    exercise_pie_chart(data)

    age_range_heartAttack_BarChart(data)


