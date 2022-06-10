import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, SelectFwe, f_classif, SelectFpr
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, GridSearchCV

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

    train_data = np.nan_to_num(train_data, nan=0)

    train_labels[train_labels < 0] = 0

    # balance_model = RandomOverSampler(sampling_strategy="minority")
    # train_data, train_labels = balance_model.fit_resample(train_data, train_labels)
    configurations = [
        # (None, True),
                      #(LinearSVC(C=0.01, penalty="l1", dual=False), False), #good
                      #(LinearSVC(C=0.02, penalty="l1", dual=False), False),
                      (LinearSVC(C=0.015, penalty="l1", dual=False), False), #good
                      (LinearSVC(C=0.008, penalty="l1", dual=False), False), #- not good
                      #(SelectKBest(f_classif, k=3), True),
                      #(SelectKBest(f_classif, k=4), True), - not good
                      #(SelectKBest(f_classif, k=5), True), -not good
                      #(SelectKBest(f_classif, k=6), True),
                      #(SelectKBest(f_classif, k=8), True), #good
                      #(SelectKBest(f_classif, k=10), True),

                      #(SelectPercentile(chi2), True),
                      #(LogisticRegression(C=1, penalty="l1", solver="liblinear"), False),
                      # (LogisticRegression(C=2, penalty="l1"), False),
                      # (LogisticRegression(C=3, penalty="l1", solver="liblinear"), False),
                      # (RandomForestClassifier(n_estimators=2), False),
                      # (RandomForestClassifier(max_depth=2, max_features=5), False), # not good
                      #(RandomForestClassifier(max_depth=3, max_features=8), False),
                      #(DecisionTreeClassifier(max_depth=2, max_features=5), False), #good
                      (DecisionTreeClassifier(max_depth=3, max_features=8), False)
                      # (XGBClassifier(max_depth=2), False),
                      # (XGBClassifier(max_depth=5), False)
                    ]
    count = 0
    for config, cond in configurations:
        count+=1
        print("Feature selection: ", config)

        if cond is True:
            train_data_c = config.fit_transform(train_data, train_labels)
            features = get_features_model(config)
            print("FEAT ", features)
            classifiy(train_data_c, train_labels, features, name="Dec")

        else:
            select_model = config.fit(train_data, train_labels)
            model = SelectFromModel(select_model, prefit=True)
            train_data_c = model.transform(train_data)
            features = get_features_model(model)
            print("FEAT ", features)
            classifiy(train_data_c, train_labels, features, name="SVC" + str(count))

    print("Pipeline DONE!")


def outliers_removal(train_data, train_labels):
    train_data = np.nan_to_num(train_data, nan=0)

    train_labels[train_labels < 0] = 0

    models = [
              IsolationForest(),
              IsolationForest(max_features=5, n_estimators=10),
              IsolationForest(n_estimators=8),
              IsolationForest(n_estimators=10)
              ]


    for isf in models:
        train_data_c, train_labels_c = checking_for_outliers(isf, train_data, train_labels)



def checking_for_outliers(isf, x, y):
    isf.fit(x, y)
    out = isf.predict(x)

    print("Out ", out)
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


def classifiy(train_data, train_labels, feat, name=""):
    train_data = np.nan_to_num(train_data, nan=0)

    train_labels[train_labels < 0] = 0
    params = {

        "max_depth": [None, 2, 3, 4],
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "min_samples_leaf": np.arange(start=2, stop=10, step=2),
        "min_samples_split": np.arange(start=2, stop=10, step=2),
        # "dt__min_weight_fraction_leaf": np.arange(start=0.0, stop=5, step=1),
        "max_features": ["sqrt", "log2"],

    }

    k_fold = KFold(n_splits=5)

    model = DecisionTreeClassifier()

    sum_scores = 0
    best_model_score = 0
    best_model_estimator = model

    for train_index, test_index in k_fold.split(train_data):

        print(len(train_data))
        Xtrain, Xtest = train_data[train_index], train_data[test_index]
        Ytrain, Ytest = train_labels[train_index], train_labels[test_index]

        balance_model = RandomOverSampler(sampling_strategy="minority")
        Xtrain, Ytrain = balance_model.fit_resample(Xtrain, Ytrain)

        print("Data after 1st balancing", Xtrain.shape, Ytrain.shape)
        search = GridSearchCV(model, params, cv=5, return_train_score=True, n_jobs=-1, verbose=1,
                              scoring="f1_macro")

        search.fit(Xtrain, Ytrain)

        print(search)
        est = search.best_estimator_
        # print("ESST", est)
        # print("BESST", search.best_score_)

        #est_pipe = make_pipeline(est)
        est.fit(Xtrain, Ytrain)
        prediction = est.predict(Xtest)
        f1_score_est = f1_score(Ytest, prediction, average="macro")
        print("f1_score_: ", f1_score_est)

        if (f1_score_est > best_model_score):
            best_model_score = f1_score_est
            best_model_estimator = est

        sum_scores = sum_scores + f1_score_est

    mean_score = sum_scores / k_fold.get_n_splits()

    hyperparameter_tuning_scores = mean_score
    best_estimators = [best_model_estimator]
    ml.visualize_tree(best_model_estimator, feat, name)
    print("best_estimators", best_estimators)
    print("ms", hyperparameter_tuning_scores)
