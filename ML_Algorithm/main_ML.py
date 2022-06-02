import Pipeline as pipe
import DataPreparation as dp
metrics = ""

if __name__ == '__main__':
    #dp.combine_metrics_from_two_instances("branch_60", "branch_60", save_metrics=False)
    #dp.compute_difference_coverage_st("default_60", "output_60", save_metrics=True)

    #dp.check_for_differences_in_classes("branch_180", "output_180")

    dp.efficiency_test_labeling("default_300", "output_300")
    #pipe.load_data()
    #mr.all_models()