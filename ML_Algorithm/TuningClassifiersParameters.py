import numpy as np
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
from subprocess import call
import os

class ClassifiersParameters:
    hyperparameter_tuning_scores = None
    best_estimators = None
    best_scores = None
    basic_scores = None
    k_fold = None


    def __init__(self, hyperparameter_tuning_scores, best_estimators, best_scores, basic_scores, k_fold):
        self.hyperparameter_tuning_scores = hyperparameter_tuning_scores
        self.best_estimators = best_estimators
        self.best_scores = best_scores
        self.basic_scores = basic_scores
        self.k_fold  = k_fold


    def tune_hyperparams(self, estimator_name, estimator, estimator_params, base_score, train_data, train_labels, k_fold):
        # DATA PREPROCESSING

        pipe = Pipeline([
            # ("pca", PCA()),
            ('LDA', LinearDiscriminantAnalysis()),
            #('SVD', TruncatedSVD()),
            (estimator_name, estimator)])
        best_model_estimator = pipe

        best_model_score = base_score
        sum_scores = 0

        for train_index, test_index in k_fold.split(train_data):

            Xtrain, Xtest = train_data[train_index], train_data[test_index]
            Ytrain, Ytest = train_labels[train_index], train_labels[test_index]



            search = GridSearchCV(pipe, estimator_params, cv=5, return_train_score=True, n_jobs=-1, verbose=1,
                                  scoring="f1_macro")

            search.fit(Xtrain, Ytrain)

            est = search.best_estimator_
            print("ESST", est)
            print("BESST", search.best_score_)

            est_pipe = make_pipeline(est)
            est_pipe.fit(Xtrain, Ytrain)
            prediction = est_pipe.predict(Xtest)

            f1_score_est = f1_score(Ytest, prediction, average="macro")
            print("f1_score_: ", f1_score_est)

            if (f1_score_est > best_model_score):
                best_model_score = f1_score_est
                best_model_estimator = est

            sum_scores = sum_scores + f1_score_est

        mean_score = sum_scores / k_fold.get_n_splits()

        print("Mean score:", mean_score, "\n")
        print("Best score:", best_model_score, "\n")
        print("Best estimator:", best_model_estimator)

        return mean_score, best_model_score, best_model_estimator


    def perform_Gaussian_model_tuning(self, models, train_data, train_labels):
        # Gaussian

        params = {
            # "pca__n_components": np.linspace(0.0, 0.99, 5)
        }

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("gaussian",
                                                                              models["GaussianNB"],
                                                                              params,
                                                                              self.basic_scores[0], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)


    def perform_Knn_model_tuning(self, models, train_data, train_labels):
        # KNN

        params = {
            # "pca__n_components": np.linspace(0.0, 0.99, 5),
            "knn__n_neighbors": np.arange(2, 5),
            "knn__weights": ["uniform", "distance"]
        }

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("knn",
                                                                              models["KNeighborsClassifier"],
                                                                              params,
                                                                              self.basic_scores[1], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, (best_model_estimator))
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)


    def perform_SVC_model_tuning(self, models, train_data, train_labels):
        # SVM
        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "svm__C": np.arange(start=1, stop=5, step=1),
                "svm__kernel": ["poly", "sigmoid"],
                "svm__random_state": [42]
            }
        ]
        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("svm",
                                                                              models["SVM"],
                                                                              params,
                                                                              self.basic_scores[2],
                                                                              train_data, train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)


    def perform_DT_model_tuning(self, models, train_data, train_labels):
        # DecisionTreeClassifier

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "dt__max_depth": np.arange(start=1, stop=5, step=1),
                "dt__min_samples_leaf": np.arange(start=1, stop=5, step=1),
                "dt__random_state": [42]
            },
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "dt__max_depth": [None],
                "dt__min_samples_leaf": np.arange(start=1, stop=5, step=1),
                "dt__random_state": [42]
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("dt",
                                                                              models["DecisionTreeClassifier"],
                                                                              params,
                                                                              self.basic_scores[3], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)


    def perform_LR_model_tuning(self, models, train_data, train_labels):
        # LogisticRegression

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "lr__C": np.arange(start=1, stop=10, step=1),
                "lr__penalty": ["l1", "l2", "none"],
                "lr__random_state": [42],
                "lr__max_iter": [1000]
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("lr",
                                                                              models["LogisticRegression"],
                                                                              params,
                                                                              self.basic_scores[4], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)


    def perform_RF_model_tuning(self, models, train_data, train_labels):
        # RandomForest

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "rf__bootstrap": [True],
                "rf__max_depth": [3, None],
                "rf__max_features": [1, 3, 10],
                "rf__min_samples_leaf": [3, 4, 5],
                "rf__min_samples_split": [8, 10],
                "rf__n_estimators": [10, 20]
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("rf",
                                                                              models["RandomForest"],
                                                                              params,
                                                                              self.basic_scores[5], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)