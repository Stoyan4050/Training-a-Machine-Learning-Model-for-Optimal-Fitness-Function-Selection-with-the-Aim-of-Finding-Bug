import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import CombineAllData as ad
import numpy as np
import VD_A as vd

ap_classes = []
ap_classes_eff = []

def statisctical_significance_test(data_1, data_2, class_given):
    # print("Wilcoxon", res)
    choose_proper_test(data_1, data_2)
    #mannwhitneyu_test(data_1, data_2, class_given)
    #wilcoxon_test(data_1, data_2, class_given)
    t_test(data_1, data_2, class_given)


    #choose_proper_test(data_1, data_2)
def wilcoxon_test(data_1, data_2, class_given):
    global ap_classes, ap_classes_eff

    res = stats.wilcoxon(data_1, data_2)
    if res[1] < 0.05:
        ap_classes.append(class_given)
        print(data_1)
        print(data_2)
        res_vd = vd.VD_A(data_1.tolist(), data_2.tolist())
        ap_classes_eff.append([class_given, res_vd[0], res_vd[1]])

    print(res)
    # print(ap_classes)
    print(len(ap_classes))

def mannwhitneyu_test(data_1, data_2, class_given):
    global ap_classes

    res = stats.mannwhitneyu(x=data_1, y=data_2, alternative='two-sided')

    if res[1] < 0.05:
        ap_classes.append(class_given)

    print(res)
    print(ap_classes)
    print(len(ap_classes))

def t_test(data_1, data_2, class_given, data_equal_size1, data_equal_size2):
    global ap_classes, ap_classes_eff


    res = stats.ttest_ind(data_1, data_2)

    if res[1] < 0.05:
        ap_classes.append(class_given)
        res_vd = vd.VD_A(data_equal_size1.tolist(), data_equal_size2.tolist())
        ap_classes_eff.append([class_given, res_vd[0], res_vd[1]])
    print(res)
    # print(ap_classes)
    print(len(ap_classes))

def choose_proper_test(data_1, data_2):
    #print(stats.levene(data_1, data_2))
    # print(shapiro_1)
    if stats.shapiro(data_1)[1] > 0.05:
        if stats.shapiro(data_2)[1] > 0.05:
            print(stats.shapiro(data_1))
            print(stats.shapiro(data_2))
            return True

    print(stats.shapiro(data_1))
    print(stats.shapiro(data_2))
    return False

def make_data_equal_size(data, size):
    count = len(data)
    data1 = list(data)
    while count < size:
        count += 1
        print("Less than 10 tests")
        data1.append(0.0)
    return np.array(data1)

def fill_empty_df(df1_class, df2_class, df1, df2, c, COLUMNS_TO_SKIP, COLUMN_TO_TEST):
    if not df1_class.empty:
        df_res1 = df1_class.drop(COLUMNS_TO_SKIP, axis=1)
        df_res1 = df_res1.drop(df1.columns[[0]], axis=1)
        data_needed_crit1 = df_res1[COLUMN_TO_TEST].values

    else:
        no_data = {}
        no_data["class"] = c
        no_data[COLUMN_TO_TEST] = 0.0
        df_res1 = pd.DataFrame(data=no_data, index=[0])
        data_needed_crit1 = df_res1[COLUMN_TO_TEST].values

        print("No data class " + c)


    if not df2_class.empty:
        df_res2 = df2_class.drop(COLUMNS_TO_SKIP, axis=1)
        df_res2 = df_res2.drop(df2.columns[[0]], axis=1)
        data_needed_crit2 = df_res2[COLUMN_TO_TEST].values

    else:
        no_data = {}
        no_data["class"] = c
        no_data[COLUMN_TO_TEST] = 0.0
        df_res2 = pd.DataFrame(data=no_data, index=[0])
        print("No data class " + c)
        data_needed_crit2 = df_res2[COLUMN_TO_TEST].values

    return data_needed_crit1, data_needed_crit2

def make_df_chosen_size(df1_class, df2_class, df1, df2, COLUMNS_TO_SKIP, COLUMN_TO_TEST, size):
    df_res1 = df1_class.drop(COLUMNS_TO_SKIP, axis=1)
    df_res1 = df_res1.drop(df1.columns[[0]], axis=1)
    data_needed_crit1 = df_res1[COLUMN_TO_TEST].values
    data_needed_crit1 = make_data_equal_size(data_needed_crit1, size)

    df_res2 = df2_class.drop(COLUMNS_TO_SKIP, axis=1)
    df_res2 = df_res2.drop(df2.columns[[0]], axis=1)
    data_needed_crit2 = df_res2[COLUMN_TO_TEST].values
    data_needed_crit2 = make_data_equal_size(data_needed_crit2, size)

    return data_needed_crit1, data_needed_crit2

def mutation_score_data(crit1, crit2, crit3):
    df = pd.read_csv("mutation_scores.csv" )
    df1 = df.loc[df['configuration'] == crit1]
    df2 = df.loc[df['configuration'] == crit2]
    df3 = df.loc[df['configuration'] == crit3]

    df1.to_csv("combined_results_mut_" + crit1 + ".csv", index=False)
    df2.to_csv("combined_results_mut_" + crit2 + ".csv", index=False)
    df3.to_csv("combined_results_mut_" + crit3 + ".csv", index=False)


def prepare_data_for_st(criteria1, criteria2, fill_till_10_test = False):
    global ap_classes, ap_classes_eff



    print("in data prep")
    COLUMNS_TO_SKIP = ["criterion", "configuration_id", "Random_Seed", "Total_Goals", "Total_Branches", "Lines",
                       "Covered_Goals", "Generations", "Statements_Executed", "Fitness_Evaluations", "Tests_Executed",
                       "Total_Time", "Size", "Result_Size", "Length", "Result_Length", "Total_Branches_Real",
                       "Coverage", "CoverageTimeline_T1", "CoverageTimeline_T2", "CoverageTimeline_T3",
                       "CoverageTimeline_T4", "CoverageTimeline_T5", "CoverageTimeline_T6"]

    #COLUMNS_TO_SKIP = ["round","killed_mutants","total_mutants"]
    COLUMN_TO_TEST = "BranchCoverage"
    #COLUMN_TO_TEST = "mutation_score_percent"


    # always put the longer file first
    df1 = pd.read_csv("combined_results_" + criteria1 + ".csv")
    df2 = pd.read_csv("combined_results_" + criteria2 + ".csv")
    # only for mutation:
    # df1 = df1.drop('configuration', axis=1)
    # df1 = df1.drop('project', axis=1)
    # df2 = df2.drop('configuration', axis=1)
    # df2 = df2.drop('project', axis=1)
    # CHANGE TARGET_CLASS to class
    # ---------------------------------------------------

    classes = ad.get_appropriate_classes(criteria1)

    for c in classes:
        df1_class = df1.loc[df1["TARGET_CLASS"] == c]

        df2_class = df2.loc[df2["TARGET_CLASS"] == c]
        # shapiro test
        data_needed_crit1, data_needed_crit2 = make_df_chosen_size(df1_class, df2_class,
                                                             df1, df2, COLUMNS_TO_SKIP, COLUMN_TO_TEST, 10)

        shapiro = choose_proper_test(data_needed_crit1, data_needed_crit1)

        if shapiro:
            # t-test
            data_needed_crit1, data_needed_crit2 = fill_empty_df(df1_class, df2_class,
                                                                 df1, df2, c,
                                                                 COLUMNS_TO_SKIP, COLUMN_TO_TEST)

            data_equal_size1, data_equal_size2 = make_df_chosen_size(df1_class, df2_class,
                                            df1, df2, COLUMNS_TO_SKIP, COLUMN_TO_TEST, 10)
            print("t-test")
            if np.array_equal(data_needed_crit1, data_needed_crit2):
                print("skip class")
            else:
                print(c)
                t_test(data_needed_crit1, data_needed_crit2, c, data_equal_size1, data_equal_size2)
        else:
            # wilcoxon-test
            data_needed_crit1, data_needed_crit2 = make_df_chosen_size(df1_class, df2_class,
                                            df1, df2, COLUMNS_TO_SKIP, COLUMN_TO_TEST, 10)

            print("wilcoxon")

            if np.array_equal(data_needed_crit1, data_needed_crit2):
                print("skip class")
            else:
                wilcoxon_test(data_needed_crit1, data_needed_crit2, c)

    df_res_eff = pd.DataFrame(ap_classes_eff, columns=["class", "A_size", "magnitude"])
    df_res_eff.to_csv("res_eff_" + criteria1 + "_" + criteria2 + "_0.csv", index=False)
    ad.compute_average_data(criteria1, ap_classes)
    ad.compute_average_data(criteria2, ap_classes)
    ad.all_metrics_for_classes_with_test_results(criteria="branch60", classes_given=ap_classes, save_metrics=True)

    #df_res_mean.reset_index(drop=True, inplace=True)
    #df_res_mean.to_csv('res_tests_' + criteria + '.csv', index=False)


