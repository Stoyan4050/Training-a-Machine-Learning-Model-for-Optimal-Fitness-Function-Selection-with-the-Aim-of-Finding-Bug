class ClassMetrics:
    # Source = https://gromit.iiar.pwr.wroc.pl/p_inf/ckjm/intro.html
    # Class name
    class_name = None

    metrics = {}

    static_criteria = ["WMC", "DIT", "NOC", "CBO", "RFC", "LCOM", "Ca", "Ce", "NPM", "LCOM3", "LOC", "DAM", "MOA", "MFA", "CAM", "IC", "CBM", "AMC"]

    def __init__(self, all_data):
        print(all_data)
        split_data = all_data.split(" ")
        len_data = len(split_data)
        len_criteria = len(self.static_criteria)

        for i in range(len_criteria):
            self.metrics[self.static_criteria[len_criteria - i - 1]] = float(split_data[len_data - i - 1])

        self.class_name = split_data[0]

    def get_metrics(self):
        return self.metrics

    def get_class_name(self):
        return self.class_name

