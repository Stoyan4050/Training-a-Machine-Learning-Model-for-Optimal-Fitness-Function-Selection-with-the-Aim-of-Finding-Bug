import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, SelectFwe, f_classif, SelectFpr

import TuningClassifiersParameters as cp

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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import export_graphviz
from sklearn.feature_selection import chi2
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest


from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler, SVMSMOTE

import matplotlib.pyplot as plt
from subprocess import call
import os
import Pipeline as pp

preprocess = ""
features = None


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

def createSinglePlot(score, labels):
    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(16, 8))
    rects1 = ax.bar(x, score, width, label="classifiers")
    ax.set_ylabel('Scores')
    ax.set_title('Performance of estimators')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()
    plt.show()


def all_models(train_data, train_labels, feature_model, features, outlier):

    models = {
        "GaussianNB": GaussianNB(),
        # "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3, weights="distance"),
        "SVM": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoost": GradientBoostingClassifier(),
        "XGBClassifier": XGBClassifier(),
        "AdaBoost" : AdaBoostClassifier()

    }

    assert "GaussianNB" in models and isinstance(models["GaussianNB"], GaussianNB), "There is no GaussianNB in models"
    assert "DecisionTreeClassifier" in models and isinstance(models["DecisionTreeClassifier"],
                                                             DecisionTreeClassifier), "There is no DecisionTreeClassifier in models"
    # assert "KNeighborsClassifier" in models and isinstance(models["KNeighborsClassifier"],
    #                                                        KNeighborsClassifier), "There is no KNeighborsClassifier in models"
    assert "SVM" in models and isinstance(models["SVM"], SVC), "There is no SVC in models"
    assert "LogisticRegression" in models and isinstance(models["LogisticRegression"],
                                                         LogisticRegression), "There is no LogisticRegression in models"

    assert "RandomForest" in models and isinstance(models["RandomForest"],
                                                         RandomForestClassifier), "There is no RandomForestClassifier in models"

    assert "GradientBoost" in models and isinstance(models["GradientBoost"],
                                                         GradientBoostingClassifier), "There is no GradientBoost in models"

    assert "XGBClassifier" in models and isinstance(models["XGBClassifier"],
                                                         XGBClassifier), "There is no XGBClassifier in models"

    assert "AdaBoost" in models and isinstance(models["AdaBoost"],
                                                         AdaBoostClassifier), "There is no AdaBoost in models"


    np.set_printoptions(threshold=np.inf)
    k_fold = KFold(n_splits=5)

    print(train_data.shape)

    #train_data, features = data_preprocessing(train_data, train_labels)
    #train_data, train_labels = checking_for_outliers(train_data, train_labels)

    print("train data after pre-processing; ", train_data.shape)


    data_balancing_models = ["smote", "rover"]

    sampl_strat = ["minority", "all"]
    rover_shrinkage = [None, 0, 1, 2]
    smote_k_neighbors = [1, 2, 3]
    for balancer in data_balancing_models:
        if balancer == "rover":
            for strat in sampl_strat:
                for shrink in rover_shrinkage:
                    pipe = pp.Pipeline(feature_model)
                    pipe.set_features(features)
                    pipe.set_outlier_model(outlier)

                    balance_model = RandomOverSampler(sampling_strategy=strat, shrinkage=shrink)
                    pipe.set_data_balancing(balance_model)
                    tuning = cp.ClassifiersParameters(hyperparameter_tuning_scores=np.empty(0),
                                                      best_estimators=np.empty(0), best_scores=np.empty(0),
                                                      oversampling_name=balancer, oversampling_model=balance_model,
                                                      k_fold=k_fold)
                    parameter_tuning(tuning, models, train_data, train_labels, pipe)

        if balancer == "smote":
            for strat in sampl_strat:
                for k_n in smote_k_neighbors:
                    pipe = pp.Pipeline(feature_model)
                    pipe.set_features(features)
                    pipe.set_outlier_model(outlier)

                    balance_model = SMOTE(sampling_strategy=strat, k_neighbors=k_n)
                    pipe.set_data_balancing(balance_model)
                    tuning = cp.ClassifiersParameters(hyperparameter_tuning_scores=np.empty(0),
                                                      best_estimators=np.empty(0), best_scores=np.empty(0),
                                                      oversampling_name=balancer, oversampling_model=balance_model,
                                                      k_fold=k_fold)

                    if check_smote(balance_model, train_data, train_labels):
                        parameter_tuning(tuning, models, train_data, train_labels, pipe)

        # Hyper Parameter Tuning


    # Get results

def parameter_tuning(tuning, models, train_data, train_labels, pipe):
    tuning.perform_Gaussian_model_tuning(models, train_data, train_labels)
    tuning.perform_SVC_model_tuning(models, train_data, train_labels)
    tuning.perform_DT_model_tuning(models, train_data, train_labels)
    tuning.perform_LR_model_tuning(models, train_data, train_labels)
    tuning.perform_RF_model_tuning(models, train_data, train_labels)
    tuning.perform_GBR_model_tuning(models, train_data, train_labels)
    tuning.perform_XGB_model_tuning(models, train_data, train_labels)
    tuning.perform_AdaBoost_model_tuning(models, train_data, train_labels)

    best_classifier, best_score = get_results_from_tuning(train_data, train_labels, tuning, features)
    pipe.set_classifier(str(best_classifier))
    pipe.set_best_score(str(best_score))

    pipe.to_csv()

def get_results_from_tuning(train_data, train_labels, tuning, features):

    # Change This
    best_estimators = tuning.best_estimators.reshape(int(tuning.best_estimators.size / 2), 2)
    # final_scores = np.empty(0)

    # print(best_estimators)
    # for model in best_estimators:
    #     sumScores = 0
    #     for train_index, test_index in tuning.k_fold.split(train_data):
    #         Xtrain, Xtest = train_data[train_index], train_data[test_index]
    #         Ytrain, Ytest = train_labels[train_index], train_labels[test_index]
    #         Xtrain, Ytrain = data_balancing(Xtrain, Ytrain)
    #         #checking_for_outliers(Xtrain, Ytrain)
    #         # Change this
    #         pipe = make_pipeline(model[0], model[1])
    #         pipe.fit(Xtrain, Ytrain)
    #         prediction = pipe.predict(Xtest)
    #
    #         score = f1_score(Ytest, prediction, average="macro")
    #         # print("Best cv score", score)
    #         sumScores = sumScores + score
    #         print("Pipe: ", pipe)
    #
    #     score = sumScores / tuning.k_fold.get_n_splits()
    #     final_scores = np.append(final_scores, score)
    #
    #     print("F1 score:", score, "\n")

    # print(basic_scores)
    # createDoublePlot(tuning.hyperparameter_tuning_scores, tuning.basic_scores, ["NaiveBayes", "KNN", "SVM", "DecTree", "LogRegr", "RandomForest", "GradientBoost", "XGB"],
    #                  "Best estimators", "Estimator with basic parameters")

    #createSinglePlot(tuning.hyperparameter_tuning_scores, ["NaiveBayes", "SVM", "DecTree", "LogRegr", "RandomForest", "GradientBoost", "XGB", "AdaBoost"])
    best_model = best_estimators[np.argmax(tuning.hyperparameter_tuning_scores)]
    print(best_model)

    print(tuning.hyperparameter_tuning_scores)
    save_results(best_model, np.max(tuning.hyperparameter_tuning_scores), "no_data")

    # visualize_tree(best_estimators[3][0], features)
    # pipe = make_pipeline(best_model[0], best_model[1])
    # pipe.fit(train_data, train_labels)

    return best_model, np.max(tuning.hyperparameter_tuning_scores)


def check_smote(model, x, y):
    try:
        print("tuk")
        x_train, y_train = model.fit_resample(x, y)
        return True
    except:
        print("tuk2")
        return False

def save_results(estimator, score, preprocess):
    df1 = pd.read_csv("ML_res.csv")
    data = {"Estimator": str(estimator), "Score": score, "Preprocess": preprocess}
    df2 = pd.DataFrame(data=data, index=[1])
    df_res = pd.concat([df1, df2], ignore_index=True)

    df_res.to_csv("ML_res.csv", index=False)

def visualize_tree(estimator, features):
    os.environ["PATH"] += os.pathsep + 'D:\\PROGRAMS\\Graphviz\\bin\\'
    print(features)
    # Export as dot file
    export_graphviz(estimator, out_file='Trees/tree.dot',

                    #feature_names=["cbo", "cboModified", "fanin", "fanout", "wmc", "dit", "noc", "rfc", "lcom", "lcom*", "tcc", "lcc", "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty", "defaultMethodsQty", "visibleMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty", "finalFieldsQty", "synchronizedFieldsQty", "nosi", "loc", "returnQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty", "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty"],
                    feature_names=features,
                    #["WMC", "DIT", "NOC", "CBO", "RFC", "LCOM", "Ca", "Ce", "NPM", "LCOM3", "LOC", "DAM", "MOA", "MFA","CAM", "IC", "CBM", "AMC"]
                    class_names=["Branch", "Branch + Output Diversity"],
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    # Convert to png using system command (requires Graphviz)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

