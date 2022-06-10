import numpy as np
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, SelectFwe, f_classif, SelectFpr

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import IsolationForest
import ML_instance_1 as ml
import Pipeline


def feature_selection(train_data, train_labels):
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

    configurations = [
        # (None, True),
                      #(LinearSVC(C=0.01, penalty="l1", dual=False), False), #good
                      #(LinearSVC(C=0.02, penalty="l1", dual=False), False),
                      #(LinearSVC(C=0.015, penalty="l1", dual=False), False), #good
                      #(LinearSVC(C=0.008, penalty="l1", dual=False), False), #- not good
                      # (SelectKBest(f_classif, k=3), True),
                      #(SelectKBest(f_classif, k=4), True), - not good
                      #(SelectKBest(f_classif, k=5), True), #-not good
                      # (SelectKBest(f_classif, k=6), True),
                      # (SelectKBest(f_classif, k=8), True), #good
                      # (SelectKBest(f_classif, k=10), True),  # good
                      # (SelectPercentile(chi2), True),
                      #(LogisticRegression(C=1, penalty="l1", solver="liblinear"), False),
                      # (LogisticRegression(C=2, penalty="l1"), False),
                      #(LogisticRegression(C=3, penalty="l1", solver="liblinear"), False),
                      #(RandomForestClassifier(n_estimators=2), False),
                      #(RandomForestClassifier(max_depth=2, max_features=5), False), # not good
                      #(RandomForestClassifier(max_depth=3, max_features=8), False),
                      #(DecisionTreeClassifier(max_depth=4, max_features=10), False),
                      #(DecisionTreeClassifier(max_depth=3, max_features=8), False),
                      #(DecisionTreeClassifier(), False),
                      (XGBClassifier(max_depth=2), False),
                      (XGBClassifier(max_depth=5), False)
    ]

    for config, cond in configurations:
        print("Feature selection: ", config)

        if config is None:
            outliers_removal(train_data, train_labels,  "None", "All")

        elif cond is True:
            train_data_c = config.fit_transform(train_data, train_labels)
            features = get_features_model(config)
            print("FEAT ", features)
            outliers_removal(train_data_c, train_labels, str(config), str(features))

        else:
            select_model = config.fit(train_data, train_labels)
            model = SelectFromModel(select_model, prefit=True)
            train_data_c = model.transform(train_data)
            features = get_features_model(model)
            print("FEAT ", features)
            outliers_removal(train_data_c, train_labels, str(model), str(features))


    print("Pipeline DONE!")


def outliers_removal(train_data, train_labels, model_features, features):
    models = [
              None,
              IsolationForest(),
              IsolationForest(max_features=5, n_estimators=10),
              IsolationForest(n_estimators=8),
              IsolationForest(n_estimators=20)
              ]

    list_pipes = []

    for isf in models:
        if isf is None:
            ml.all_models(train_data, train_labels, model_features, features, "None")

        else:
            train_data_c, train_labels_c = checking_for_outliers(isf, train_data, train_labels)
            ml.all_models(train_data_c, train_labels_c, model_features, features, str(isf))

    return


def checking_for_outliers(isf, x, y):
    isf.fit(x, y)
    out = isf.predict(x)

    y_new = []
    x_new = []
    for i in range(len(out)):
        if out[i] != -1:
            y_new.append(y[i])
            x_new.append(x[i])

    return np.array(x_new), np.array(y_new)



def get_features_model(model):
    feature_names_in_ = ["cbo", "cboModified", "fanin", "fanout", "wmc", "dit", "noc", "rfc", "lcom", "lcom*", "tcc",
                         "lcc", "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty",
                         "protectedMethodsQty", "defaultMethodsQty", "visibleMethodsQty", "abstractMethodsQty",
                         "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty",
                         "publicFieldsQty", "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty",
                         "finalFieldsQty", "synchronizedFieldsQty", "nosi", "loc", "returnQty", "loopQty",
                         "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty",
                         "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty",
                         "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers",
                         "logStatementsQty"]

    feature_names_in_ = np.array(feature_names_in_)
    feature_names_in_.reshape(1, 49)

    return model.get_feature_names_out(feature_names_in_)


def extract_features(df1, train_data):
    features = set()
    final_features = set()
    init_flag = 0
    df = df1.iloc[:, 2:]
    # print(df)
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


