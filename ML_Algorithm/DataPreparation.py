import pandas as pd
import numpy as np

def compute_difference_coverage(criteria1, criteria2, save_metrics=False):
    df1 = pd.read_csv("res_tests_" + criteria1 + ".csv")
    df2 = pd.read_csv("res_tests_" + criteria2 + ".csv")

    df_res = pd.DataFrame([])

    for i in range(len(df1.index)):
        curr_df2 = df2.loc[df2["TARGET_CLASS"] == df1.iloc[i]["TARGET_CLASS"]]
        if not curr_df2.empty:
            df1_perf = df1.iloc[i][df1.columns[1]] # out
            df2_perf = curr_df2.iloc[0][curr_df2.columns[1]] # branch

            difference = df1_perf - df2_perf
            label = 0

            if difference > 0:
                label = 1
            elif difference < 0:
                label = -1

            data = {"TARGET_CLASS": df1.iloc[i]["TARGET_CLASS"], "Difference": label}
            df_diff = pd.DataFrame(data=data, index=[0])
            df_res = pd.concat([df_res, df_diff], ignore_index=True)

    if save_metrics:
        df_res.to_csv('results_difference_' + criteria1 + "_" + criteria2 + '.csv', index=False)


def compute_difference_coverage_st(criteria1, criteria2, save_metrics=False):
    df1 = pd.read_csv("StatisticalTestResults/MutationDefault/res_tests_tw_od_mut_" + criteria1 + ".csv")
    df2 = pd.read_csv("StatisticalTestResults/MutationDefault/res_tests_tw_od_mut_" + criteria2 + ".csv")
    df_metrics = pd.read_csv("StatisticalTestResults/MutationDefault/metrics_chosen_classes_tw_mut_default60.csv")

    df_res = pd.DataFrame([])

    for i in range(len(df1.index)):
        curr_df2 = df2.loc[df2["class"] == df1.iloc[i]["class"]]
        if not curr_df2.empty:
            df1_perf = df1.iloc[i][df1.columns[1]] # branch
            df2_perf = curr_df2.iloc[0][curr_df2.columns[1]] # out

            difference = df1_perf - df2_perf

            if difference > 0:
                label = -1
                data = {"class": df1.iloc[i]["class"], "Difference": label}
                df_diff = pd.DataFrame(data=data, index=[0])
                df_res = pd.concat([df_res, df_diff], ignore_index=True)

            elif difference < 0:
                label = 1
                data = {"class": df1.iloc[i]["class"], "Difference": label}
                df_diff = pd.DataFrame(data=data, index=[0])
                df_res = pd.concat([df_res, df_diff], ignore_index=True)
            else:
                df_metrics.drop(df_metrics.index[df_metrics['class'] == df1.iloc[i]["class"]], inplace=True)

    if save_metrics:
        df_metrics.to_csv('metrics_tw_mut_' + criteria1 + "_" + criteria2 + '.csv', index=False)
        df_res.to_csv('results_difference_tw_mut_' + criteria1 + "_" + criteria2 + '.csv', index=False)




def combine_metrics_from_two_instances(criteria1, criteria2, save_metrics=False):
    df1 = pd.read_csv("metrics_chosen_classes_" + criteria1 + ".csv")
    df2 = pd.read_csv("metrics_chosen_classes_" + criteria2 + ".csv")

    df_res = pd.DataFrame([])

    for i in range(len(df1.index)):
        curr_df2 = df2.loc[df2["class"] == df1.iloc[i]["class"]]
        if not curr_df2.empty:
            df_res = pd.concat([df_res, curr_df2], ignore_index=True)

    if save_metrics:
        df_res.to_csv('combine_metrics_' + criteria1 + "_" + criteria2 + '.csv', index=False)



def efficiency_test_labeling(criteria1, criteria2):
    # if 0, out < 0.5 elif 1 out > 0.5
    df1 = pd.read_csv("EfficiencyTest/res_eff_" + criteria1 + "_" + criteria2 + "_0.csv")
    #sdf1 = df1.reset_index()  # make sure indexes pair with number of rows
    dict_res = {}
    df_res_eff = pd.DataFrame([])

    for index, row in df1.iterrows():
        if row["A_size"] < 0.5:
            dict_res["class"] = row["class"]
            dict_res["label"] = 1
            dict_res["magnitude"] = row["magnitude"]

            df_res = pd.DataFrame(dict_res, index=[0])
            df_res_eff = pd.concat([df_res_eff, df_res], ignore_index=True)

        elif row["A_size"] > 0.5:
            dict_res["class"] = row["class"]
            dict_res["label"] = -1
            dict_res["magnitude"] = row["magnitude"]

            df_res = pd.DataFrame(dict_res, index=[0])
            df_res_eff = pd.concat([df_res_eff, df_res], ignore_index=True)

    df_res_eff.to_csv("res_eff_label_" + criteria1 + "_" + criteria2 + ".csv", index=False)

def check_for_differences_in_classes(criteria1, criteria2):
    df1 = pd.read_csv('combine_metrics_' + criteria1 + "_" + criteria2 + '.csv')
    df2 = pd.read_csv('results_difference_' + criteria1 + "_" + criteria2 + '.csv')

    for i in range(len(df1.index)):
        if df2.iloc[i]["TARGET_CLASS"] != df1.iloc[i]["class"]:
            print("Problem")

