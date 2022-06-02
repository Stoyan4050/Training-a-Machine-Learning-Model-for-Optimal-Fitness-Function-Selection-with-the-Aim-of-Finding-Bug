import ClassMetrics as ClassMetrics
import pandas as pd
import os

def create_csv_from_class_metrics():
    file = open('Others/all_metrics.txt', 'r')
    lines = file.readlines()
    combined_metrics_data = []
    for line in lines:
        class_metrics = ClassMetrics.ClassMetrics(line)
        class_name = class_metrics.get_class_name()
        combined_metrics_data.append({"class_name": class_name} | class_metrics.get_metrics())

    df = pd.DataFrame(data=combined_metrics_data)
    df.to_csv('all_metrics.csv')


def get_appropriate_classes(criteria):
    df = pd.read_csv("combined_results_" + criteria + ".csv")
    all_classes = set()
    # 1 for branch, 2 for mutation
    classes = df.iloc[:, 1].values.tolist()
    for class1 in classes:
        all_classes.add(class1)

    all_classes = sorted(list(all_classes))
    return all_classes


def all_metrics_for_classes_with_test_results(criteria, classes_given=None, save_metrics=False):
    df = pd.read_csv("class_metrics.csv")

    if classes_given is None:
        classes = get_appropriate_classes(criteria)
    else:
        classes = classes_given
    # df.iloc[:, 1].values.tolist()
    no_data_classes = []
    df_res = pd.DataFrame([])
    counter = 0
    for c in classes:
        counter += 1
        curr = df.loc[df["class"] == c]
        if curr.empty:
            no_data_classes.append(c)
        df_res = pd.concat([df_res, curr], ignore_index=True)

    if save_metrics:
        df_res1 = df_res.drop(df.columns[[0]], axis=1)
        df_res1.to_csv('metrics_chosen_classes_tw_eff_' + criteria + '.csv', index=False)

    return no_data_classes


def compute_average_data(criteria, classes_given=None):
    COLUMNS_TO_SKIP = ["criterion", "configuration_id", "Random_Seed", "Total_Goals", "Total_Branches", "Lines",
                       "Covered_Goals", "Generations", "Statements_Executed", "Fitness_Evaluations", "Tests_Executed",
                       "Total_Time", "Size", "Result_Size", "Length", "Result_Length", "Total_Branches_Real",
                       "Coverage", "CoverageTimeline_T1", "CoverageTimeline_T2", "CoverageTimeline_T3",
                       "CoverageTimeline_T4", "CoverageTimeline_T5", "CoverageTimeline_T6"]
    COLUMNS_TO_AVG = ["BranchCoverage"]
    #COLUMNS_TO_SKIP = ["round","killed_mutants","total_mutants"]
    #COLUMNS_TO_AVG = ["mutation_score_percent"]

    df = pd.read_csv("combined_results_" + criteria + ".csv")

    # only for mutation:
    #df = df.drop('configuration', axis=1)
    #df = df.drop('project', axis=1)
    # ---------------------------------------------------

    if classes_given is None:
        classes = get_appropriate_classes(criteria)
    else:
        classes = classes_given

    df_res_mean = pd.DataFrame([])

    for c in classes:
        df1_class = df.loc[df["class"] == c]
        if not df1_class.empty:
            df_res1 = df1_class.drop(COLUMNS_TO_SKIP, axis=1)
            df_res1 = df_res1.drop(df.columns[[0]], axis=1)
            mean_values = {}
            mean_values["class"] = c
            for crit in COLUMNS_TO_AVG:
                # Median here
                kk2 = df_res1[crit]
                count = len(kk2)
                while count < 10:
                    count+=1
                    print("Less than 10 tests")
                    kk2.loc[count] = 0.0

                mean_values[crit] = kk2.median()

            df_mean = pd.DataFrame(data=mean_values, index=[0])
            df_res_mean = pd.concat([df_res_mean, df_mean])
        else:
            mean_values = {}
            mean_values["class"] = c
            for crit in COLUMNS_TO_AVG:
                mean_values[crit] = 0.0

            df_mean = pd.DataFrame(data=mean_values, index=[0])
            df_res_mean = pd.concat([df_res_mean, df_mean])

    df_res_mean.reset_index(drop=True, inplace=True)
    df_res_mean.to_csv('res_tests_tw_ob_eff_' + criteria + '.csv', index=False)


def remove_common_el_lists(list1, list2):
    res_list = []
    for el in list1:
        if not el in list2:
            res_list.append(el)

    return res_list
