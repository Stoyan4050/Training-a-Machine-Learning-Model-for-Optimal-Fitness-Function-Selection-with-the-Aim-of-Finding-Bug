import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

basic_scores = np.empty(0)

# Array that holds the best set of parameters for each model
best_estimators = np.empty(0)

# Array that holds the mean score obtained during hyperparamter tuning for each model
hyperparameter_tuning_scores = np.empty(0)

# Array containing the score of the highest rated estimator for each model
best_scores = np.empty(0)

# The K-fold cross validator we are going to use
k_fold = KFold(n_splits=10)


def createDoublePlot(score1, score2, labels, leg1, leg2):
    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(16, 8))
    rects1 = ax.bar(x - width / 2, score1, width, label=leg1)
    rects2 = ax.bar(x + width / 2, score2, width, label=leg2)
    ax.set_ylabel('Scores')
    ax.set_title('Performance of estimators')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()
    plt.show()


def tune_hyperparams(estimator_name, estimator, estimator_params, base_score, train_data, train_labels, k_fold):
    best_model_estimator = Pipeline([("pca", PCA()), (estimator_name, estimator)])
    best_model_score = base_score
    sum_scores = 0

    for train_index, test_index in k_fold.split(train_data):

        Xtrain, Xtest = train_data[train_index], train_data[test_index]
        Ytrain, Ytest = train_labels[train_index], train_labels[test_index]

        pipe = Pipeline([("pca", PCA()), (estimator_name, estimator)])
        search = GridSearchCV(pipe, estimator_params, cv=10, return_train_score=True, n_jobs=-1, verbose=1,
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


def convert_data(data):
    list = []
    for l in data:
        list.append(l[0].astype('float'))

    return list


def all_models():
    global basic_scores, best_estimators, hyperparameter_tuning_scores, best_scores, k_fold

    models = {
        "GaussianNB": GaussianNB(),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3, weights="distance"),
        "SVM": SVC(C=10, kernel="poly", random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=42),
        "LogisticRegression": LogisticRegression(C=10, random_state=42, penalty="none", max_iter=50000)

    }

    assert "GaussianNB" in models and isinstance(models["GaussianNB"], GaussianNB), "There is no GaussianNB in models"
    assert "DecisionTreeClassifier" in models and isinstance(models["DecisionTreeClassifier"],
                                                             DecisionTreeClassifier), "There is no DecisionTreeClassifier in models"
    assert "KNeighborsClassifier" in models and isinstance(models["KNeighborsClassifier"],
                                                           KNeighborsClassifier), "There is no KNeighborsClassifier in models"
    assert "SVM" in models and isinstance(models["SVM"], SVC), "There is no SVC in models"
    assert "LogisticRegression" in models and isinstance(models["LogisticRegression"],
                                                         LogisticRegression), "There is no LogisticRegression in models"

    train_data = np.genfromtxt("combine_metrics_output_60_branch_60.csv", delimiter=',')[1:, 1:]
    train_labels = np.array(
        convert_data(np.genfromtxt("results_difference_output_60_branch_60.csv", delimiter=',')[1:, 1:])).astype(int)

    np.set_printoptions(threshold=np.inf)
    # print(train_data.shape)
    # print(len(train_labels))

    basic_parameters_algorithm(models, train_data, train_labels)

    # Hyper Parameter Tuning
    perform_Gaussian_model_tuning(models, train_data, train_labels)
    perform_Knn_model_tuning(models, train_data, train_labels)
    perform_SVC_model_tuning(models, train_data, train_labels)
    perform_DT_model_tuning(models, train_data, train_labels)
    perform_LR_model_tuning(models, train_data, train_labels)

    # Get results
    get_results_from_tuning(train_data, train_labels)


def basic_parameters_algorithm(models, train_data, train_labels):
    global basic_scores

    names = np.empty(0)

    for name, model in models.items():

        print(name, ":")
        sumScores = 0
        names = np.append(names, name)

        for train_index, test_index in k_fold.split(train_data):
            Xtrain, Xtest = train_data[train_index], train_data[test_index]
            Ytrain, Ytest = train_labels[train_index], train_labels[test_index]

            pipe = make_pipeline(PCA(), model)
            pipe.fit(Xtrain, Ytrain)
            prediction = pipe.predict(Xtest)

            score = f1_score(Ytest, prediction, average="macro")
            print(name, "    ", score)
            sumScores = sumScores + score

        score = sumScores / k_fold.get_n_splits()
        basic_scores = np.append(basic_scores, score)
        print("F1 score:", score, "\n")

    print(basic_scores)
    print(names)


def perform_Gaussian_model_tuning(models, train_data, train_labels):
    global hyperparameter_tuning_scores, best_estimators, best_scores
    # Gaussian
    # 8x8
    params = {
        "pca__n_components": np.linspace(0.0, 0.99, 100)
    }

    mean_score, best_model_score, best_model_estimator = tune_hyperparams("gaussian",
                                                                          models["GaussianNB"],
                                                                          params,
                                                                          basic_scores[0],
                                                                          train_data, train_labels, k_fold)

    hyperparameter_tuning_scores = np.append(hyperparameter_tuning_scores, mean_score)
    best_estimators = np.append(best_estimators, best_model_estimator)
    best_scores = np.append(best_scores, best_model_score)

    print("best_estimators", best_estimators)
    print("ms", hyperparameter_tuning_scores)
    print("bs", best_scores)


def perform_Knn_model_tuning(models, train_data, train_labels):
    global hyperparameter_tuning_scores, best_estimators, best_scores
    # KNN
    # 8x8
    params = {
        "pca__n_components": np.linspace(0.0, 0.99, 100),
        "knn__n_neighbors": np.arange(2, 100),
        "knn__weights": ["uniform", "distance"]
    }

    mean_score, best_model_score, best_model_estimator = tune_hyperparams("knn",
                                                                          models["KNeighborsClassifier"],
                                                                          params,
                                                                          basic_scores[1], train_data,
                                                                          train_labels, k_fold)

    hyperparameter_tuning_scores = np.append(hyperparameter_tuning_scores, mean_score)
    best_estimators = np.append(best_estimators, best_model_estimator)
    best_scores = np.append(best_scores, best_model_score)

    print("be", best_estimators)
    print("ms", hyperparameter_tuning_scores)
    print("bs", best_scores)


def perform_SVC_model_tuning(models, train_data, train_labels):
    global hyperparameter_tuning_scores, best_estimators, best_scores
    # SVC
    # 8x8
    params = [
        {
            "pca__n_components": np.linspace(0.0, 0.99, 50),
            "svm__C": np.arange(start=1, stop=100, step=1),
            "svm__kernel": ["poly", "sigmoid"],
            "svm__random_state": [42]
        }
    ]
    mean_score, best_model_score, best_model_estimator = tune_hyperparams("svm",
                                                                          models["SVM"],
                                                                          params,
                                                                          basic_scores[2],
                                                                          train_data, train_labels, k_fold)

    hyperparameter_tuning_scores = np.append(hyperparameter_tuning_scores, mean_score)
    best_estimators = np.append(best_estimators, best_model_estimator)
    best_scores = np.append(best_scores, best_model_score)

    print("be", best_estimators)
    print("ms", hyperparameter_tuning_scores)
    print("bs", best_scores)


def perform_DT_model_tuning(models, train_data, train_labels):
    global hyperparameter_tuning_scores, best_estimators, best_scores
    # DecisionTreeClassifier
    # 8x8

    params = [
        {
            "pca__n_components": np.linspace(0.0, 0.99, 50),
            "dt__max_depth": np.arange(start=1, stop=50, step=1),
            "dt__min_samples_leaf": np.arange(start=1, stop=50, step=1),
            "dt__random_state": [42]
        },
        {
            "pca__n_components": np.linspace(0.0, 0.99, 50),
            "dt__max_depth": [None],
            "dt__min_samples_leaf": np.arange(start=1, stop=50, step=1),
            "dt__random_state": [42]
        }
    ]

    mean_score, best_model_score, best_model_estimator = tune_hyperparams("dt",
                                                                          models["DecisionTreeClassifier"],
                                                                          params,
                                                                          basic_scores[3], train_data,
                                                                          train_labels, k_fold)

    hyperparameter_tuning_scores = np.append(hyperparameter_tuning_scores, mean_score)
    best_estimators = np.append(best_estimators, best_model_estimator)
    best_scores = np.append(best_scores, best_model_score)

    print("be", best_estimators)
    print("ms", hyperparameter_tuning_scores)
    print("bs", best_scores)


def perform_LR_model_tuning(models, train_data, train_labels):
    global hyperparameter_tuning_scores, best_estimators, best_scores
    # LogisticRegression
    # 8x8

    params = [
        {
            "pca__n_components": np.linspace(0.0, 0.99, 50),
            "lr__C": np.arange(start=1, stop=50, step=0.2),
            "lr__penalty": ["l1", "l2", "none"],
            "lr__random_state": [42],
            "lr__max_iter": [1000]
        }
    ]

    mean_score, best_model_score, best_model_estimator = tune_hyperparams("lr",
                                                                          models["LogisticRegression"],
                                                                          params,
                                                                          basic_scores[4], train_data,
                                                                          train_labels, k_fold)

    hyperparameter_tuning_scores = np.append(hyperparameter_tuning_scores, mean_score)
    best_estimators = np.append(best_estimators, best_model_estimator)
    best_scores = np.append(best_scores, best_model_score)

    print("be", best_estimators)
    print("ms", hyperparameter_tuning_scores)
    print("bs", best_scores)


def get_results_from_tuning(train_data, train_labels):
    global best_estimators, basic_scores

    best_estimators = best_estimators.reshape(int(best_estimators.size / 2), 2)
    # print(best_estimators)
    final_scores = np.empty(0)

    for model in (best_estimators):
        sumScores = 0
        print(model)
        for train_index, test_index in k_fold.split(train_data):
            Xtrain, Xtest = train_data[train_index], train_data[test_index]
            Ytrain, Ytest = train_labels[train_index], train_labels[test_index]

            pipe = make_pipeline(model[0], model[1])
            pipe.fit(Xtrain, Ytrain)
            prediction = pipe.predict(Xtest)

            score = f1_score(Ytest, prediction, average="macro")
            print("Best cv score", score)
            sumScores = sumScores + score

        score = sumScores / k_fold.get_n_splits()
        final_scores = np.append(final_scores, score)
        print("Average Best Score: ", score)
        print("F1 score:", score, "\n")

    print(basic_scores)
    createDoublePlot(final_scores, basic_scores, ["NaiveBayes", "KNN", "SVM", "DecTree", "LogRegr"],
                     "Best estimators", "Estimator with basic parameters")

    best_model = best_estimators[np.argmax(hyperparameter_tuning_scores)]
    print(best_model)
    pipe = make_pipeline(best_model[0], best_model[1])
    pipe.fit(train_data, train_labels)
    # prediction = pipe.predict(mnist_28x28_test)
