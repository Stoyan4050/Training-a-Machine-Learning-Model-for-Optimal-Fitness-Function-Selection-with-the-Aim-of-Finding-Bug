from ML_Algorithm import ML_instance_1 as ml
from ML_Algorithm import DataPreparation as dp
from ML_Algorithm import ML_Regression as mr
metrics = ""

if __name__ == '__main__':
    #dp.combine_metrics_from_two_instances("branch_60", "branch_60", save_metrics=False)
    #dp.compute_difference_coverage_st("branch_60", "output_60", save_metrics=True)

    #dp.check_for_differences_in_classes("output_60", "default_60")
    #ml.all_models()
    mr.all_models()