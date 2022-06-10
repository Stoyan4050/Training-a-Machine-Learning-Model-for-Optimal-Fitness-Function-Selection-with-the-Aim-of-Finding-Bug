import CombineAllData as ad
import ExtractTestResults as tr
import StatisticalTests as st

metrics = ""

if __name__ == '__main__':
    #cm.find_jars("D:\Delft_1\Y3\RP_repo\projects")
    # tr.combine_all_csv_results("D:\Delft_1\Y3\Research_Project_Git_Repo\\results", criteria="default_180", save=True)
    # tr.combine_all_csv_results("D:\Delft_1\Y3\Research_Project_Git_Repo\\results", criteria="output_180", save=True)
    # tr.combine_all_csv_results("D:\Delft_1\Y3\Research_Project_Git_Repo\\results", criteria="branch_180", save=True)

    # Get all class metrics in csv
    #ad.create_csv_from_class_metrics()
    # ad.all_metrics_for_classes_with_test_results(criteria="output_60", save_metrics=True)
    #ad.compute_average_data(criteria="branch_60")
    #st.prepare_data_for_st(criteria1="output_180", criteria2="default_180")
    #st.mutation_score_data("branch_180", "default_180", "output_180")
    # 0 for branch, out out < 0.5
    # 1 fot out, branch out > 0.5

    #st.prepare_data_for_st(criteria1="default_300", criteria2="output_300")
    ad.filter_data_metrics_eff_test(criteria1="mut_branch_60", criteria2="mut_output_60")

    #st.prepare_data_for_st(criteria1="default_60", criteria2="output_60")