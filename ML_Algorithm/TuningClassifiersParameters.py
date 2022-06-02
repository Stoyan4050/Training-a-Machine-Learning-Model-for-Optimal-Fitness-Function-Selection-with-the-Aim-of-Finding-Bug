import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score, accuracy_score

from imblearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline

import matplotlib.pyplot as plt


class ClassifiersParameters:
    hyperparameter_tuning_scores = None
    best_estimators = None
    best_scores = None
    basic_scores = None
    k_fold = None
    oversampling_name = None
    oversampling_model = None

    def __init__(self, hyperparameter_tuning_scores, best_estimators, best_scores, oversampling_name,
                 oversampling_model, k_fold):
        self.hyperparameter_tuning_scores = hyperparameter_tuning_scores
        self.best_estimators = best_estimators
        self.best_scores = best_scores
        self.k_fold = k_fold
        self.oversampling_name = oversampling_name
        self.oversampling_model = oversampling_model

    def data_balancing(self, x, y):
        x_train, y_train = self.oversampling_model.fit_resample(x, y)

        return x_train, y_train


    def tune_hyperparams(self, estimator_name, estimator, estimator_params, train_data, train_labels, k_fold):
        pipe = Pipeline([
            (self.oversampling_name, self.oversampling_model),
            (estimator_name, estimator)])

        best_model_estimator = pipe

        best_model_score = 0
        sum_scores = 0

        for train_index, test_index in k_fold.split(train_data):

            Xtrain, Xtest = train_data[train_index], train_data[test_index]
            Ytrain, Ytest = train_labels[train_index], train_labels[test_index]

            Xtrain, Ytrain = self.data_balancing(Xtrain, Ytrain)

            print("Data after 1st balancing", Xtrain.shape, Ytrain.shape)
            search = GridSearchCV(pipe, estimator_params, cv=5, return_train_score=True, n_jobs=-1, verbose=2,
                                  scoring="f1_macro")

            search.fit(Xtrain, Ytrain)

            est = search.best_estimator_
            # print("ESST", est)
            # print("BESST", search.best_score_)

            est_pipe = make_pipeline(est)
            est_pipe.fit(Xtrain, Ytrain)
            prediction = est_pipe.predict(Xtest)

            f1_score_est = f1_score(Ytest, prediction, average="macro")
            # print("f1_score_: ", f1_score_est)

            if (f1_score_est > best_model_score):
                best_model_score = f1_score_est
                best_model_estimator = est

            sum_scores = sum_scores + f1_score_est

        mean_score = sum_scores / k_fold.get_n_splits()

        print("Mean score:", mean_score, "\n")
        # print("Best score:", best_model_score, "\n")
        print("Best estimator:", best_model_estimator)

        return mean_score, best_model_score, best_model_estimator

    def perform_Gaussian_model_tuning(self, models, train_data, train_labels):
        # Gaussian

        params = [{}]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("gaussian",
                                                                                   models["GaussianNB"],
                                                                                   params, train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        # print("bs", self.best_scores)

    def perform_SVC_model_tuning(self, models, train_data, train_labels):
        # SVM
        params = [{
            "svm__C":[0.1, 1, 10, 20, 100, 150, 200],
            "svm__kernel": ["poly", "sigmoid", "rbf"],
            "svm__gamma": ["scale"
               # , "auto"
                           ]
        }]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("svm",
                                                                                   models["SVM"],
                                                                                   params,

                                                                                   train_data, train_labels,
                                                                                   self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        # print("bs", self.best_scores)

    def perform_DT_model_tuning(self, models, train_data, train_labels):
        # DecisionTreeClassifier

        params = {
            #     {
            #         "dt__max_depth": np.arange(start=1, stop=20, step=1),
            #         "dt__criterion": ["gini", "entropy"],
            #         "dt__splitter": ["best", "random"],
            #         "dt__min_samples_leaf": np.arange(start=1, stop=20, step=1),
            #         "dt__min_samples_split": np.arange(start=1, stop=20, step=1),
            #         "dt__min_weight_fraction_leaf": np.arange(start=0.0, stop=5, step=0.5),
            #         "dt__max_features": ["auto", "sqrt", "log2"],
            #         "dt__max_leaf_nodes": np.arange(start=1, stop=20, step=1),
            #         "dt__min_impurity_decrease": np.arange(start=0.0, stop=5, step=0.5),
            #         "dt__class_weight":  [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]
            #
            # },
            "dt__max_depth": [None, 2, 3, 4],
            "dt__criterion": ["gini", "entropy"],
            "dt__splitter": ["best", "random"],
            "dt__min_samples_leaf": np.arange(start=2, stop=10, step=2),
            "dt__min_samples_split": np.arange(start=2, stop=10, step=2),
            # "dt__min_weight_fraction_leaf": np.arange(start=0.0, stop=5, step=1),
            "dt__max_features": ["sqrt", "log2"],
            # "dt__max_leaf_nodes": np.arange(start=1, stop=20, step=1),
            # "dt__min_impurity_decrease": np.arange(start=0.0, stop=5, step=0.5),
            # "dt__class_weight": [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]

        }

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("dt",
                                                                                   models["DecisionTreeClassifier"],
                                                                                   params,
                                                                                   train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        print("bs", self.best_scores)

    def perform_LR_model_tuning(self, models, train_data, train_labels):
        # LogisticRegression

        params = [{
            "lr__C": np.arange(start=1, stop=10, step=1),
            "lr__penalty": ["l2", "none"],
            "lr__random_state": [42],
            "lr__max_iter": [1000]
        }]


        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("lr",
                                                                                   models["LogisticRegression"],
                                                                                   params,
                                                                                   train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        # print("bs", self.best_scores)

    def perform_RF_model_tuning(self, models, train_data, train_labels):
        # RandomForest

        params = [{
            "rf__bootstrap": [True],
            "rf__max_depth": [3, None],
            "rf__max_features": [1, 3, 10],
            "rf__min_samples_leaf": [3, 4, 5],
            "rf__min_samples_split": [8, 10],
            "rf__n_estimators": [10, 20]
        }]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("rf",
                                                                                   models["RandomForest"],
                                                                                   params,
                                                                                   train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        # print("bs", self.best_scores)

    def perform_GBR_model_tuning(self, models, train_data, train_labels):

        params = [{
            "gbr__max_features": [3, None],
            # "gbr__max_depth": [3, 8, None],
            "gbr__max_leaf_nodes": [2, 3, 10],
            # "gbr__min_samples_leaf": [3, 4, 5],
            "gbr__min_samples_split": [8, 10],
            "gbr__min_weight_fraction_leaf": [0, 0.25, 0.5]
        }]

        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("gbr",
                                                                                   models["GradientBoost"],
                                                                                   params,
                                                                                   train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        est = best_model_estimator
        self.best_estimators = np.append(self.best_estimators, est)
        # Pipeline(steps=[('gbr',GradientBoostingRegressor(max_features=3, max_leaf_nodes=3, min_samples_split=10, min_weight_fraction_leaf=0.5))]))

        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        # print("bs", self.best_scores)

    def perform_XGB_model_tuning(self, models, train_data, train_labels):
        # XGB

        params = [{
                # "xgb__min_child_weight": [0, 1, 2, 10],
                "xgb__max_depth": [3, 6, 10],
                # "xgb__sampling_method": ["uniform", "subsample", "gradient_based"],
                # "xgb__tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"],
                # "xgb__max_bin": [256, 512],
                "xgb__num_parallel_tree": [1, 2, 3]
            }]


        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("xgb",
                                                                                   models["XGBClassifier"],
                                                                                   params,
                                                                                   train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, best_model_estimator)
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        # print("bs", self.best_scores)

    def perform_AdaBoost_model_tuning(self, models, train_data, train_labels):
        # Ada

        params = [{
            "ada__n_estimators": [2, 5, 10, 20, 30, 50, 75, 100],
            "ada__learning_rate": [0.1, 0.5, 0.8, 1],
            "ada__algorithm": ["SAMME", "SAMME.R"]
        }]



        mean_score, best_model_score, best_model_estimator = self.tune_hyperparams("ada",
                                                                                   models["AdaBoost"],
                                                                                   params,
                                                                                   train_data,
                                                                                   train_labels, self.k_fold)

        self.hyperparameter_tuning_scores = np.append(self.hyperparameter_tuning_scores, mean_score)
        self.best_estimators = np.append(self.best_estimators, (best_model_estimator))
        self.best_scores = np.append(self.best_scores, best_model_score)

        print("best_estimators", self.best_estimators)
        print("ms", self.hyperparameter_tuning_scores)
        # print("bs", self.best_scores)


