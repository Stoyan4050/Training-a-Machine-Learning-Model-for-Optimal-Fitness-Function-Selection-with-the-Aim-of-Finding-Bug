from Data_Extraction import CombineAllData as ad
import os

metrics = ""

if __name__ == '__main__':
    #cm.find_jars("D:\Delft_1\Y3\RP_repo\projects")
    #tr.combine_all_csv_results("D:\Delft_1\Y3\RP_repo\\results", criteria="branch_60")
    # Get all class metrics in csv
    #ad.create_csv_from_class_metrics()
    ad.all_metrics_for_classes_with_test_results(criteria="branch_60", save_metrics=True)
    ad.compute_average_data(criteria="branch_60")