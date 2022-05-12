import numpy as np
import pandas as pd
import RegressionTuning as rt

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
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
from subprocess import call
import os

class RegressionParameters:
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
        self.k_fold = k_fold


    def tune_hyperparams(self, estimator_name, estimator, estimator_params, base_score, train_data, train_labels, k_fold):
        # DATA PREPROCESSING

        pipe = Pipeline([
            #("pca", PCA()),
            #('LDA', LinearDiscriminantAnalysis()),
            #('SVD', TruncatedSVD()),

            (estimator_name, estimator)])
        best_model_estimator = pipe

        best_model_score = base_score
        sum_scores = 0

        for train_index, test_index in k_fold.split(train_data):

            Xtrain, Xtest = train_data[train_index], train_data[test_index]
            Ytrain, Ytest = train_labels[train_index], train_labels[test_index]



            search = GridSearchCV(pipe, estimator_params, cv=5, return_train_score=True, n_jobs=-1, verbose=1,
                                  scoring="r2")

            search.fit(Xtrain, Ytrain)

            est = search.best_estimator_
            print("ESST", est)
            print("BESST", search.best_score_)

            est_pipe = make_pipeline(est)
            est_pipe.fit(Xtrain, Ytrain)
            prediction = est_pipe.predict(Xtest)

            r2 = metrics.r2_score(Ytest, prediction)
            print("R2: ", r2)

            if (r2 > best_model_score):
                best_model_score = r2
                best_model_estimator = est

            sum_scores = sum_scores + r2

        mean_score = sum_scores / k_fold.get_n_splits()

        print("Mean score:", mean_score, "\n")
        print("Best score:", best_model_score, "\n")
        print("Best estimator:", best_model_estimator)

        return mean_score, best_model_score, best_model_estimator



    # models = {
    #     "BayesianRidge": BayesianRidge(),
    #     "LinearRegression": LinearRegression(),
    #     "SVR": SVR(),
    #     "DecisionTreeRegressor": DecisionTreeRegressor(),
    #     "GradientBoostingRegressor": GradientBoostingRegressor(),
    #     "RandomForestRegressor": RandomForestRegressor(),
    #     "KernelRidge": KernelRidge(),
    #     "ElasticNet": ElasticNet(),
    #     "XGBRegressor": XGBRegressor()
    # }



    def perform_BayesianRidge_model_tuning(self, models, train_data, train_labels):
        # Gaussian

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "bayesian__n_iter": [1000],
                "bayesian__tol": [1e-3, 1e-2],
                "bayesian__fit_intercept": [True, False],
                #"bayesian__normalize": [True, False],
                "bayesian__alpha_1": [1e-6, 1e-3],
                "bayesian__lambda_1": [1e-6, 1e-3],
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("bayesian",
                                                                              models["BayesianRidge"],
                                                                              params,
                                                                              self.basic_scores[0], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)


    def perform_LR_model_tuning(self, models, train_data, train_labels):

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                #"lr__normalize": [True, False],
                "lr__fit_intercept": [True, False],
                "lr__copy_X": [True, False],
                "lr__positive": [True, False],
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("lr",
                                                                              models["LinearRegression"],
                                                                              params,
                                                                              self.basic_scores[1], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, (best_model_estimator))
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)


    def perform_SVR_model_tuning(self, models, train_data, train_labels):
        # SVM
        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "svr__C": np.arange(start=1, stop=5, step=1),
                #"svr__kernel": ["poly", "sigmoid", "linear", "rbf"],
                #"svr__degree": [1, 3, 5],
                # "svr__gamma": ["scale", "auto"],
                #"svr__epsilon": [0.1, 0.001, 0.5]
            }
        ]
        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("svr",
                                                                              models["SVR"],
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
                                                                              models["DecisionTreeRegressor"],
                                                                              params,
                                                                              self.basic_scores[3], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)


    def perform_GBR_model_tuning(self, models, train_data, train_labels):

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "gbr__max_features": [3, None],
                # "gbr__max_depth": [3, 8, None],
                "gbr__max_leaf_nodes": [1, 3, 10],
                #"gbr__min_samples_leaf": [3, 4, 5],
                "gbr__min_samples_split": [8, 10],
                "gbr__min_weight_fraction_leaf": [0, 0.25, 0.5]
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("gbr",
                                                                              models["GradientBoostingRegressor"],
                                                                              params,
                                                                              self.basic_scores[4], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        print(self.best_estimators)
        est = best_model_estimator
        self.best_estimators = np.append(self.best_estimators, est)
        #Pipeline(steps=[('gbr',GradientBoostingRegressor(max_features=3, max_leaf_nodes=3, min_samples_split=10, min_weight_fraction_leaf=0.5))]))
        print(self.best_estimators)

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
                                                                              models["RandomForestRegressor"],
                                                                              params,
                                                                              self.basic_scores[5], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)
        #self.visualize_tree(best_model_estimator[1][0])

    def perform_KR_model_tuning(self, models, train_data, train_labels):
        # RandomForest

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "kr__alpha": [1, 1.5, 2, 3, 5],
                # "kr__kernel": ,
                "kr__gamma": [1, 1.5, 2, 3, None],
                "kr__degree": [1, 3, 4, 5],
                "kr__coef0": [1, 2, 3],
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("kr",
                                                                              models["KernelRidge"],
                                                                              params,
                                                                              self.basic_scores[6], train_data,
                                                                              train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)

    def perform_EN_model_tuning(self, models, train_data, train_labels):
        # RandomForest

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                "en__l1_ratio": [0, 0.4, 0.8, 1],
                #"en__normalize": [True, False],
                "en__fit_intercept": [True],
                "en__max_iter": [5000],
                # "en__selection": ["cyclic", "random"],
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("en",
                                                                                   models["ElasticNet"],
                                                                                   params,
                                                                                   self.basic_scores[7], train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)

    def perform_XGB_model_tuning(self, models, train_data, train_labels):
        # RandomForest

        params = [
            {
                # "pca__n_components": np.linspace(0.0, 0.99, 5),
                # "xgb__min_child_weight": [0, 1, 2, 10],
                "xgb__max_depth": [3, 6, 10],
                #"xgb__sampling_method": ["uniform", "subsample", "gradient_based"],
                #"xgb__tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"],
                "xgb__max_bin": [256, 512],
                # "xgb__num_parallel_tree": [1, 2, 3]
            }
        ]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("xgb",
                                                                                   models["XGBRegressor"],
                                                                                   params,
                                                                                   self.basic_scores[8], train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)

    def visualize_tree(self, estimator):
        os.environ["PATH"] += os.pathsep + 'D:\\PROGRAMS\\Graphviz\\bin\\'
        # Export as dot file
        export_graphviz(estimator, out_file='tree_regr.dot',
                        #feature_names=["WMC", "DIT", "NOC", "CBO", "RFC", "LCOM", "Ca", "Ce", "NPM", "LCOM3", "LOC", "DAM", "MOA", "MFA","CAM", "IC", "CBM", "AMC"],
                        feature_names=["Feature_1", "Feature_2"],
                        class_names=["-1", "0", "1"],
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        # Convert to png using system command (requires Graphviz)
        call(['dot', '-Tpng', 'tree_regr.dot', '-o', 'tree_regr.png', '-Gdpi=600'])