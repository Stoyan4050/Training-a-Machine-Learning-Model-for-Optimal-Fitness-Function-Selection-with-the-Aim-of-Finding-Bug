import numpy as np
import Preprocessing as pp
import pandas as pd

class Pipeline:
    features = ""
    feat_selection = ""
    outlier_model = ""
    best_score = ""
    classifier = ""
    data_balancing = ""

    def __init__(self, feat_select):
        self.feat_selection = feat_select

    def set_features(self, features):
        self.features = features

    def set_outlier_model(self, outlier):
        self.outlier_model = outlier

    def set_classifier(self, classifier):
        self.classifier = classifier

    def set_best_score(self, score):
        self.best_score = score

    def set_data_balancing(self, data_balancing):
        self.data_balancing = data_balancing

    def to_csv(self):
        df = pd.read_csv("Model_performance.csv")
        df_dict = {}
        df_dict["FEATURES_SELECTION"] = self.feat_selection
        df_dict["OUTLIER_REMOVAL"] = self.outlier_model
        df_dict["DATA_BALANCING"] = self.data_balancing
        df_dict["CLASSIFIER"] = self.classifier
        df_dict["SCORE"] = self.best_score
        df_dict["FEATURES"] = self.features

        df_new = pd.DataFrame(data=df_dict, index=[0])

        df_res = pd.concat([df, df_new])
        df_res.to_csv('Model_performance.csv', index=False)

def convert_data(data):
    list = []
    for l in data:
        list.append(l[0].astype('float'))
    return list


def load_data():
    train_data = np.genfromtxt("ReadyForML/metrics_twbranch_60_output_60.csv", delimiter=',')[1:, 2:]
    train_labels = np.array(
        convert_data(
            np.genfromtxt("ReadyForML/results_difference_twbranch_60_output_60.csv", delimiter=',')[1:, 1:])).astype(int)

    # train_data = np.genfromtxt("ReadyForML/metrics_twdefault_300_output_300.csv", delimiter=',')[1:, 1:]
    # train_labels = np.array(
    #     convert_data(np.genfromtxt("ReadyForML/results_difference_twdefault_300_output_300.csv", delimiter=',')[1:, 1:])).astype(int)

    train_data = np.nan_to_num(train_data, nan=0)

    train_labels[train_labels < 0] = 0

    print("Loaded data", train_data.shape)

    df = pd.read_csv("Model_performance.csv")
    #df = df[0:0]
    df.to_csv('Model_performance.csv', index=False)

    pp.feature_selection(train_data, train_labels)
