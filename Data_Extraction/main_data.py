import CombineAllData as ad
import ExtractTestResults as tr
import StatisticalTests as st

metrics = ""

if __name__ == '__main__':
    #cm.find_jars("D:\Delft_1\Y3\RP_repo\projects")
    tr.combine_all_csv_results("D:\Delft_1\Y3\Research_Project_Git_Repo\\results", criteria="default_300", save=True)
    # Get all class metrics in csv
    #ad.create_csv_from_class_metrics()
    # ad.all_metrics_for_classes_with_test_results(criteria="output_60", save_metrics=True)
    #ad.compute_average_data(criteria="branch_60")
    #st.prepare_data_for_st(criteria1="default_300", criteria2="output_300")
    #st.prepare_data_for_st(criteria1="default_60", criteria2="output_60")