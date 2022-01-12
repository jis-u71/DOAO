import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from scipy.stats import mode

from preprocessing import *

import warnings
warnings.filterwarnings(action='ignore')

np.random.seed(0)


def select_competent_classifier(X_trainval_scaled, y_trainval, inner_kfold):
    # cross validated committees (cvc)
    mlp_best_score = 0
    for hidden_layer_size in [tuple([i]) for i in range(3, 21)]:
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation='tanh', max_iter=300)
        mlp_scores = cross_validate(mlp, X_trainval_scaled, y_trainval, scoring='accuracy', cv=inner_kfold, return_estimator=True)
        mlp_score = mlp_scores['test_score'].mean()

        if mlp_score > mlp_best_score:
            mlp_best_score = mlp_score
            mlp_best_models = mlp_scores['estimator']
            mlp_best_hyperparameters = {'hidden_layer_sizes': hidden_layer_size}

    dt_best_score = 0
    for min_samples_leaf in [1,2,3,5]:
        for min_samples_split in [5,10]:
            dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
            dt_scores = cross_validate(dt, X_trainval_scaled, y_trainval, scoring='accuracy', cv=inner_kfold, return_estimator=True)
            dt_score = dt_scores['test_score'].mean()

            if dt_score > dt_best_score:
                dt_best_score = dt_score
                dt_best_models = dt_scores['estimator']
                dt_best_hyperparameters = {'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split}

    knn_best_score = 0
    for n_neighbor in [1,3,5,7,10,20,30]:
        knn = KNeighborsClassifier(n_neighbors=n_neighbor)
        knn_scores = cross_validate(knn, X_trainval_scaled, y_trainval, scoring='accuracy', cv=inner_kfold, return_estimator=True)
        knn_score = knn_scores['test_score'].mean()

        if knn_score > knn_best_score:
            knn_best_score = knn_score
            knn_best_models = knn_scores['estimator']
            knn_best_hyperparameters = {'n_neighbors': n_neighbor}

    lda_best_score = 0
    lda = LinearDiscriminantAnalysis()
    lda_scores = cross_validate(lda, X_trainval_scaled, y_trainval, scoring='accuracy', cv=inner_kfold, return_estimator=True)
    lda_score = lda_scores['test_score'].mean()

    if lda_score > lda_best_score:
        lda_best_score = lda_score
        lda_best_models = lda_scores['estimator']

    lr_best_score = 0
    lr = LogisticRegression()
    lr_scores = cross_validate(lr, X_trainval_scaled, y_trainval, scoring='accuracy', cv=inner_kfold, return_estimator=True)
    lr_score = lr_scores['test_score'].mean()

    if lr_score > lr_best_score:
        lr_best_score = lr_score
        lr_best_models = lr_scores['estimator']
    
    svm_best_score = 0        
    for gamma in [2**i for i in range(-5, 6)]:
        for C in [2**i for i in range(-3, 11)]:
            svm = SVC(kernel='rbf', C=C, gamma=gamma)
            svm_scores = cross_validate(svm, X_trainval_scaled, y_trainval, scoring='accuracy', cv=inner_kfold, return_estimator=True)
            svm_score = svm_scores['test_score'].mean()

            if svm_score > svm_best_score:
                svm_best_score = svm_score
                svm_best_models = svm_scores['estimator']
                svm_best_hyperparameters = {'C': C, 'gamma': gamma}
    
    # candidate classifier list
    candidate_classifier_list = np.array([mlp_best_models, dt_best_models, knn_best_models, lda_best_models, lr_best_models, svm_best_models])
    
    best_score_val = [mlp_best_score, dt_best_score, 
                        knn_best_score, lda_best_score, 
                        lr_best_score, svm_best_score]

    print("best score on val : ", best_score_val)

    best_score = np.max(best_score_val)

    best_idices = np.where(best_score_val==best_score)[0]
    print("best indices : ", best_idices)

    best_idx = best_idices[np.random.randint(len(best_idices))]
    print("best idx : ", best_idx)

    # best cvc
    competent_classifier = candidate_classifier_list[best_idx]
    print("competent classifier : ", competent_classifier)

    return competent_classifier


def eval_phase(X_test_scaled, y_test, competent_classifier_list, outer_idx, n_iter):
    predictions = np.empty((len(X_test_scaled), len(competent_classifier_list)))

    for idx, cvc in enumerate(competent_classifier_list):
        predictions_cvc = np.empty((len(X_test_scaled), len(cvc)))
        
        for clf_idx, clf in enumerate(cvc):
            y_test_hats = clf.predict(X_test_scaled)
            predictions_cvc[:, clf_idx] = y_test_hats
            
        majority_voting_in_cvc = mode(predictions_cvc, axis=1)[0]
        predictions[:, idx] = np.ravel(majority_voting_in_cvc)
                
    # majority voting
    majority_voting = mode(predictions, axis=1)[0]
    final_prediction = np.ravel(majority_voting)

    print("y test : ", y_test)
    print("final prediction : ", final_prediction)
    print("# of final prediciton : ", len(final_prediction))

    # calculate error rate
    num_errors = np.sum(y_test != final_prediction)
    error_rate = num_errors / len(y_test) * 100
    print("### iter : {} === {}th num error : {}\n".format(n_iter, outer_idx, num_errors))

    return error_rate