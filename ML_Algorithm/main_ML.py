import ML_instance_1 as ml
import DataPreparation as dp
import ML_Regression as mr
metrics = ""

if __name__ == '__main__':
    #dp.combine_metrics_from_two_instances("branch_60", "branch_60", save_metrics=False)
    #dp.compute_difference_coverage_st("default_300", "output_300", save_metrics=True)

    #dp.check_for_differences_in_classes("branch_60", "output_60")
    ml.all_models()
    #mr.all_models()