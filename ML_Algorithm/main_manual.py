import numpy as np
import manual_analysis_tools as mat

def convert_data(data):
    list = []
    for l in data:
        list.append(l[0].astype('float'))
    return list

if __name__ == '__main__':

    train_data = np.genfromtxt("ReadyForML/AfterEffTest/Default/metrics_chosen_classes_eff_output_180_default_180.csv", delimiter=',')[1:, 2:]
    train_labels = np.array(
        convert_data(
            np.genfromtxt("ReadyForML/AfterEffTest/Default/eff_no_magn_output_180_default_180.csv", delimiter=',')[1:, 1:])).astype(int)

    print(train_data)
    print(train_labels)

    #mat.classifiy(train_data, train_labels)
    #mat.outliers_removal(train_data, train_labels)
    mat.feature_selection(train_data, train_labels)