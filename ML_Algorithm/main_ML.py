from ML_Algorithm import ML_instance_1 as ml
from ML_Algorithm import DataPreparation as dp
metrics = ""

if __name__ == '__main__':
    #dp.compute_difference_coverage("output_60", "default_60", save_metrics=False)
    #dp.combine_metrics_from_two_instances("output_60", "default_60", save_metrics=True)
    #dp.check_for_differences_in_classes("output_60", "default_60")
    ml.all_models()