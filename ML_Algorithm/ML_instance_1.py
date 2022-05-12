import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, SelectFwe, f_classif, SelectFpr

import TuningClassifiersParameters as cp

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import export_graphviz
from sklearn.feature_selection import chi2

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
from subprocess import call
import os

preprocess = ""
basic_scores = np.empty(0)
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

def data_preprocessing(train_data, train_labels):
    global preprocess
    train_data_c = train_data

    # sc = StandardScaler()
    # train_data_c = sc.fit_transform(train_data)
    # preprocess = preprocess + " " + "StandardScaler "

    # pca = PCA()
    # train_data_c = pca.fit_transform(train_data, train_labels)
    # preprocess = preprocess + " " + "PCA "

    # tsne = TSNE()
    # tsne_results = tsne.fit_transform(train_data)
    # train_data_c = tsne_results
    # preprocess = preprocess + " " + "TSNE "

    # model = SelectKBest(f_classif, k=5)
    # train_data_c = model.fit_transform(train_data, train_labels)

    # model = SelectPercentile(chi2)
    # train_data_c = model.fit_transform(train_data, train_labels)

    lsvc = LinearSVC(penalty="l1", dual=False).fit(train_data, train_labels)
    model = SelectFromModel(lsvc, prefit=True)
    train_data_c = model.transform(train_data)

    # clf = ExtraTreesClassifier(n_estimators=3)
    # clf = clf.fit(train_data, train_labels)
    # model = SelectFromModel(clf, prefit=True)
    # train_data_c = model.transform(train_data)

    # forest = RandomForestClassifier(random_state=0)
    # forest.fit(train_data, train_labels)
    # model = SelectFromModel(forest, prefit=True)
    # train_data_c = model.transform(train_data)

    feature_names_in_ = ["cbo", "cboModified", "fanin", "fanout", "wmc", "dit", "noc", "rfc", "lcom", "lcom*", "tcc", "lcc", "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty", "defaultMethodsQty", "visibleMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty", "finalFieldsQty", "synchronizedFieldsQty", "nosi", "loc", "returnQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty", "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty"],
    feature_names_in_ = np.array(feature_names_in_[0])
    feature_names_in_.reshape(1, 49)


    print("FEATURES", model.get_feature_names_out(feature_names_in_))


    return train_data_c, model.get_feature_names_out(feature_names_in_)

def extract_features(df1, train_data):
    features = set()
    final_features = set()
    init_flag = 0
    df = df1.iloc[: , 2:]
    #print(df)
    df1.keys()
    for index, row in df.iterrows():
        for i, column in enumerate(row):
            if column in train_data[index][:]:
                features.add(df.columns[i])

        if init_flag == 0:
            final_features = features
            init_flag = 1
        else:
            if final_features != features:
                print("problem")
                final_features = features

    return final_features

def convert_data(data):
    list = []
    for l in data:
        list.append(l[0].astype('float'))

    return list


def all_models():
    models = {
        "GaussianNB": GaussianNB(),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3, weights="distance"),
        "SVM": SVC(C=10, kernel="poly", random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=42),
        "LogisticRegression": LogisticRegression(C=10, random_state=42, penalty="none", max_iter=10000),
        "RandomForest": RandomForestClassifier(random_state=42, max_depth=None, min_samples_leaf=2)
    }
    #print(models["RandomForest"].get_params().keys())

    assert "GaussianNB" in models and isinstance(models["GaussianNB"], GaussianNB), "There is no GaussianNB in models"
    assert "DecisionTreeClassifier" in models and isinstance(models["DecisionTreeClassifier"],
                                                             DecisionTreeClassifier), "There is no DecisionTreeClassifier in models"
    assert "KNeighborsClassifier" in models and isinstance(models["KNeighborsClassifier"],
                                                           KNeighborsClassifier), "There is no KNeighborsClassifier in models"
    assert "SVM" in models and isinstance(models["SVM"], SVC), "There is no SVC in models"
    assert "LogisticRegression" in models and isinstance(models["LogisticRegression"],
                                                         LogisticRegression), "There is no LogisticRegression in models"

    assert "RandomForest" in models and isinstance(models["RandomForest"],
                                                         RandomForestClassifier), "There is no RandomForestClassifier in models"

    train_data = np.genfromtxt("ReadyForML/metrics_twbranch_60_output_60.csv", delimiter=',')[1:, 2:]
    train_labels = np.array(
        convert_data(np.genfromtxt("ReadyForML/results_difference_twbranch_60_output_60.csv", delimiter=',')[1:, 1:])).astype(int)

    # train_data = np.genfromtxt("ReadyForML/metrics_twdefault_300_output_300.csv", delimiter=',')[1:, 1:]
    train_data = np.nan_to_num(train_data, nan=0)
    # train_labels = np.array(
    #     convert_data(np.genfromtxt("ReadyForML/results_difference_twdefault_300_output_300.csv", delimiter=',')[1:, 1:])).astype(int)

    np.set_printoptions(threshold=np.inf)

    k_fold = KFold(n_splits=5)

    basic_scores = basic_parameters_algorithm(models, train_data, train_labels, k_fold)


    train_data, features = data_preprocessing(train_data, train_labels)

    print("FEAT",features)


    tuning = cp.ClassifiersParameters(hyperparameter_tuning_scores=np.empty(0),
                                      best_estimators=np.empty(0), best_scores=np.empty(0),
                                      basic_scores=basic_scores, k_fold=k_fold)



    # Hyper Parameter Tuning
    tuning.perform_Gaussian_model_tuning(models, train_data, train_labels)
    tuning.perform_Knn_model_tuning(models, train_data, train_labels)
    tuning.perform_SVC_model_tuning(models, train_data, train_labels)
    tuning.perform_DT_model_tuning(models, train_data, train_labels)
    tuning.perform_LR_model_tuning(models, train_data, train_labels)
    tuning.perform_RF_model_tuning(models, train_data, train_labels)

    # Get results
    get_results_from_tuning(train_data, train_labels, tuning, features)


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

            score = f1_score(Ytest, prediction, average="macro")
            print(name, "    ", score)
            sumScores = sumScores + score

        score = sumScores / k_fold.get_n_splits()
        basic_scores = np.append(basic_scores, score)
        print("F1 score:", score, "\n")

        # if name  == "DecisionTreeClassifier":
        #     #visualize_tree(model)
        #     tree.plot_tree(model)
        #     plt.show()

    print(basic_scores)
    print(names)
    return basic_scores

def get_results_from_tuning(train_data, train_labels, tuning, features):
    global preprocess

    # Change This
    best_estimators = tuning.best_estimators.reshape(int(tuning.best_estimators.size / 1), 1)
    final_scores = np.empty(0)

    # print(best_estimators)
    for model in best_estimators:
        sumScores = 0
        for train_index, test_index in tuning.k_fold.split(train_data):
            Xtrain, Xtest = train_data[train_index], train_data[test_index]
            Ytrain, Ytest = train_labels[train_index], train_labels[test_index]

            # Change this
            pipe = make_pipeline(model[0])
            pipe.fit(Xtrain, Ytrain)
            prediction = pipe.predict(Xtest)

            score = f1_score(Ytest, prediction, average="macro")
            # print("Best cv score", score)
            sumScores = sumScores + score
            print("Pipe: ", pipe)

        score = sumScores / tuning.k_fold.get_n_splits()
        final_scores = np.append(final_scores, score)

        print("F1 score:", score, "\n")

    # print(basic_scores)
    createDoublePlot(tuning.hyperparameter_tuning_scores, tuning.basic_scores, ["NaiveBayes", "KNN", "SVM", "DecTree", "LogRegr", "RandomForest"],
                     "Best estimators", "Estimator with basic parameters")

    # print(best_estimators)

    best_model = best_estimators[np.argmax(tuning.hyperparameter_tuning_scores)]
    print(final_scores)
    print(best_model)

    print(tuning.hyperparameter_tuning_scores)
    save_results(best_model, np.max(tuning.hyperparameter_tuning_scores), preprocess)

    print("Loading_Model")
    visualize_tree(best_estimators[3][0], features)
    pipe = make_pipeline(best_model[0])
    pipe.fit(train_data, train_labels)

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
    export_graphviz(estimator, out_file='tree.dot',

    #feature_names=["cbo", "cboModified", "fanin", "fanout", "wmc", "dit", "noc", "rfc", "lcom", "lcom*", "tcc", "lcc", "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty", "defaultMethodsQty", "visibleMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty", "finalFieldsQty", "synchronizedFieldsQty", "nosi", "loc", "returnQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty", "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty"],
    feature_names=features,
    #["WMC", "DIT", "NOC", "CBO", "RFC", "LCOM", "Ca", "Ce", "NPM", "LCOM3", "LOC", "DAM", "MOA", "MFA","CAM", "IC", "CBM", "AMC"]
    class_names=["Branch", "Branch + Output Diversity"],
    rounded=True, proportion=False,
    precision=2, filled=True)

    # Convert to png using system command (requires Graphviz)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

