import numpy as np
import pandas as pd
import ML_Algorithm.RegressionTuning as rt

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

preprocess = ""
basic_scores = np.empty(0)


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

def data_preprocessing(train_data):
    global preprocess
    train_data_c = train_data

    # pca = PCA()
    # train_data_c = pca.fit_transform(train_data)
    # preprocess = preprocess + " " + "PCA "

    # tsne = TSNE()
    # tsne_results = tsne.fit_transform(train_data)
    # train_data_c = tsne_results
    # preprocess = preprocess + " " + "TSNE "

    # sc = StandardScaler()
    # train_data_c = sc.fit_transform(train_data)
    # preprocess = preprocess + " " + "StandardScaler "


    return train_data_c


def convert_data(data):
    list = []
    for l in data:
        list.append(l[0].astype('float'))

    return list


def all_models():
    models = {
        "BayesianRidge": BayesianRidge(),
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "KernelRidge": KernelRidge(),
        "ElasticNet": ElasticNet(),
        "XGBRegressor": XGBRegressor()
    }

    train_data = np.genfromtxt("ReadyForML/metrics_twdefault_300_output_300.csv", delimiter=',')[1:, 1:]
    train_data = train_data[1:, 1:]
    train_data = np.nan_to_num(train_data, nan=0)
    train_labels = np.array(
        convert_data(np.genfromtxt("ReadyForML/results_difference_twdefault_300_output_300.csv", delimiter=',')[1:, 1:])).astype(int)

    np.set_printoptions(threshold=np.inf)
    print(train_data)
    print(train_labels)


    train_data = data_preprocessing(train_data)

    k_fold = KFold(n_splits=5)

    basic_scores = basic_parameters_algorithm(models, train_data, train_labels, k_fold)

    tuning = rt.RegressionParameters(hyperparameter_tuning_scores=np.empty(0),
                                      best_estimators=np.empty(0), best_scores=np.empty(0),
                                      basic_scores=basic_scores, k_fold=k_fold)



    # Hyper Parameter Tuning
    tuning.perform_BayesianRidge_model_tuning(models, train_data, train_labels)
    tuning.perform_LR_model_tuning(models, train_data, train_labels)
    tuning.perform_SVR_model_tuning(models, train_data, train_labels)
    tuning.perform_DT_model_tuning(models, train_data, train_labels)
    tuning.perform_GBR_model_tuning(models, train_data, train_labels)
    tuning.perform_RF_model_tuning(models, train_data, train_labels)
    tuning.perform_KR_model_tuning(models, train_data, train_labels)
    tuning.perform_EN_model_tuning(models, train_data, train_labels)
    tuning.perform_XGB_model_tuning(models, train_data, train_labels)

    # Get results
    get_results_from_tuning(train_data, train_labels, tuning)


def basic_parameters_algorithm(models, train_data, train_labels, k_fold):
    names = np.empty(0)
    basic_scores = np.empty(0)

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

            score = metrics.r2_score(Ytest, prediction, )
            print(name, "    ", score)
            sumScores = sumScores + score

        score = sumScores / k_fold.get_n_splits()
        basic_scores = np.append(basic_scores, score)
        print("R2 score:", score, "\n")

    print(basic_scores)
    print(names)
    return basic_scores

def get_results_from_tuning(train_data, train_labels, tuning):
    global preprocess

    # Change This
    best_estimators = tuning.best_estimators.reshape(int(tuning.best_estimators.size / 2), 2)
    final_scores = np.empty(0)

    # print(best_estimators)
    print(len(best_estimators))
    for model in best_estimators:
        sumScores = 0
        for train_index, test_index in tuning.k_fold.split(train_data):
            Xtrain, Xtest = train_data[train_index], train_data[test_index]
            Ytrain, Ytest = train_labels[train_index], train_labels[test_index]

            # Change this
            pipe = make_pipeline(model[0], model[1])
            pipe.fit(Xtrain, Ytrain)
            prediction = pipe.predict(Xtest)

            score = metrics.r2_score(Ytest, prediction)
            # print("Best cv score", score)
            sumScores = sumScores + score
            print("Pipe: ", pipe)

        score = sumScores / tuning.k_fold.get_n_splits()
        final_scores = np.append(final_scores, score)

        print("R2 score:", score, "\n")

    # print(basic_scores)

    createDoublePlot(tuning.hyperparameter_tuning_scores, tuning.basic_scores, ["Bayesian", "LR", "SVR", "DecTree", "GBR", "RandomForest",
                                                                                "KernelRidge", "ElasticNet", "XGBRegressor"],
                     "Best estimators", "Estimator with basic parameters")

    # print(best_estimators)

    best_model = best_estimators[np.argmax(tuning.hyperparameter_tuning_scores)]
    print(final_scores)
    print(best_model)

    print(tuning.hyperparameter_tuning_scores)
    save_results(best_model, np.max(tuning.hyperparameter_tuning_scores), preprocess)

    # print("Loading_Model")
    # visualize_tree(best_estimators[-1][1][1], train_labels)
    # pipe = make_pipeline(best_model[0], best_model[1])
    # pipe.fit(train_data, train_labels)

def save_results(estimator, score, preprocess):
    df1 = pd.read_csv("ML_res_regr.csv")
    data = {"Estimator": str(estimator), "Score": score, "Preprocess": preprocess}
    df2 = pd.DataFrame(data=data, index=[1])
    df_res = pd.concat([df1, df2], ignore_index=True)

    df_res.to_csv("ML_res_regr.csv", index=False)


