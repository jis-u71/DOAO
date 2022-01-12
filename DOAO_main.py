import pandas as pd
import numpy as np

from preprocessing import *
from DOAO_function import *

np.random.seed(0)
    
"""
main
"""

# import dataset
zoo = pd.read_csv("./dataset/zoo/zoo.data", header=None)
zoo.name = "zoo"
iris = pd.read_csv("./dataset/iris/iris.data", header=None)
iris.name = "iris"
wine = pd.read_csv("./dataset/wine/wine.data", header=None)
wine.name = "wine"
glass = pd.read_csv("./dataset/glass/glass.data", header=None)
glass.name = "glass"
balance = pd.read_csv("./dataset/balance/balance-scale.data", header=None)
balance.name = "balance"

# dataset_list = [zoo, iris, wine, glass, balance]
dataset_list = [iris]

# DOAO main
final_mean = {}
final_std = {}
total_error_rate = []

for dataset in dataset_list:
    error_rate_per_dataset = []

    for n_iter in range(5):
        if dataset.name == "zoo":
            cv=3
        else:
            cv=5

        # preprocessing
        X, y = make_X_y(dataset=dataset, dataset_name=dataset.name)

        # nested cross validation
        print("cv : ", cv)
        
        inner_kfold = StratifiedKFold(n_splits=cv, shuffle=True)
        outer_kfold = StratifiedKFold(n_splits=cv, shuffle=True)

        competent_classifier_list = []
        error_rate_list = []

        for outer_idx, (trainval_idx, test_idx) in enumerate(outer_kfold.split(X, y)):
            X_trainval = X[trainval_idx]
            y_trainval = y[trainval_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            # scaling
            X_trainval_scaled, X_test_scaled = scaling(X_trainval, X_test)

            # dij
            X_class_pair_list, y_class_pair_list = make_class_pair_dataset(X_trainval_scaled, y_trainval)

            for X_class_pair, y_class_pair in zip(X_class_pair_list, y_class_pair_list):
                print("\n### class pair : {}".format(np.unique(y_class_pair)))

                # select competent classifier per class pair
                competent_classifier = select_competent_classifier(X_trainval_scaled, y_trainval, inner_kfold)
                competent_classifier_list.append(competent_classifier)

            error_rate = eval_phase(X_test_scaled, y_test, competent_classifier_list, outer_idx, n_iter)
            error_rate_list.append(error_rate)
        
        mean_error_rate = np.mean(error_rate_list)
        print("\n=== iter : {}, error rate : {}".format(n_iter, mean_error_rate))

        error_rate_per_dataset.extend(error_rate_list)
        
    total_error_rate.append(error_rate_per_dataset)
    mean_of_total_error_rate = np.mean(error_rate_per_dataset)
    std_of_total_error_rate = np.std(error_rate_per_dataset)

    print("\ntotal error rate list : ", total_error_rate)

    final_mean[dataset.name] = mean_of_total_error_rate
    final_std[dataset.name] = std_of_total_error_rate

print("final mean : ", final_mean)
print("final std : ", final_std)