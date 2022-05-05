import os
import pandas as pd


def find_all_csv_results(path, criteria):
    rootdir = path + "\\" + criteria
    csv_files = []
    print("Start...")
    for project in os.listdir(rootdir):
        d = os.path.join(rootdir, project)
        if os.path.isdir(d):
            for class1 in os.listdir(d):
                d1 = os.path.join(d, class1)
                if os.path.isdir(d1):
                    d2 = d1 + "\\reports"
                    if os.path.isdir(d2):
                        for results in os.listdir(d2):
                            d3 = os.path.join(d2, results)
                            for stats in os.listdir(d3):
                                if stats.endswith((".csv")):
                                    d4 = os.path.join(d3, stats)
                                    csv_files.append(d4)

    return csv_files


def combine_all_csv_results(path, criteria, save=False):
    all_csv_files = find_all_csv_results(path, criteria)
    if save:
        df = pd.concat(map(pd.read_csv, all_csv_files), ignore_index=True)
        df.to_csv('combined_results_' + criteria + '.csv')

    print("DONE")
