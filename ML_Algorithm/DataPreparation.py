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



def combine_metrics_from_two_instances(criteria1, criteria2, save_metrics=False):
    df1 = pd.read_csv("metrics_chosen_classes_" + criteria1 + ".csv")
    df2 = pd.read_csv("metrics_chosen_classes_" + criteria2 + ".csv")

    df_res = pd.DataFrame([])

    for i in range(len(df1.index)):
        curr_df2 = df2.loc[df2["class_name"] == df1.iloc[i]["class_name"]]
        if not curr_df2.empty:
            df_res = pd.concat([df_res, curr_df2], ignore_index=True)

    if save_metrics:
        df_res.to_csv('combine_metrics_' + criteria1 + "_" + criteria2 + '.csv', index=False)


def check_for_differences_in_classes(criteria1, criteria2):
    df1 = pd.read_csv('combine_metrics_' + criteria1 + "_" + criteria2 + '.csv')
    df2 = pd.read_csv('results_difference_' + criteria1 + "_" + criteria2 + '.csv')

    for i in range(len(df1.index)):
        if df2.iloc[i]["TARGET_CLASS"] != df1.iloc[i]["class_name"]:
            print("Problem")

