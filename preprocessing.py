import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from itertools import combinations


# preprocessing

def make_X_y(dataset, dataset_name):
    if dataset_name == "zoo" or dataset_name == "glass":
        X = dataset.iloc[:, 1:-1]
        y = dataset.iloc[:, -1]

    elif dataset_name == "iris":
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

    elif dataset_name == "wine" or dataset_name == "balance":
        X = dataset.iloc[:, 1:]
        y = dataset.iloc[:, 0]

    else:
        print("Try again")

    X = X.to_numpy()
    y = y.to_numpy()

    # label encoding - y
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print("###  {}  ###".format(dataset_name))
    # print(y)

    return X, y

def scaling(X_train, X_test):
    # minmax scaling - X
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

# make class pair dataset (dij)
def make_class_pair_dataset(X, y):
    classes = np.unique(y)
    class_pair = list(combinations(classes, 2))
    
    X_class_pair_list = []
    y_class_pair_list = []

    for pair in class_pair:
        print("class pair ({}, {})".format(pair[0], pair[1]))

        first_c_idx = np.where(y == pair[0])
        second_c_idx = np.where(y == pair[1])

        #print(y[first_c_idx].shape)
        #print(y[second_c_idx].shape)

        X_class_pair = np.vstack((X[first_c_idx], X[second_c_idx]))
        y_class_pair = np.hstack((y[first_c_idx], y[second_c_idx]))

        #print(y_class_pair)
        # print(y_class_pair.shape)
        
        X_class_pair_list.append(X_class_pair)
        y_class_pair_list.append(y_class_pair)

    return X_class_pair_list, y_class_pair_list