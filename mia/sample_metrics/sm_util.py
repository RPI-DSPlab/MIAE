import json
import os


def avg_result(dir, file_suf='.json', roundToInt=False):
    """
    This function is used to average the results of an datapoint hardness metric across different seeds.
    :param dir: directory of the results
    :param file_suf: suffix of the result files (default: .json)
    :return: a dictionary of averaged results
    """

    pd_dict_list = []
    file_list = os.listdir(dir)
    file_names = [file for file in file_list if file.endswith(file_suf)]

    if len(file_names) == 0:
        raise Exception("No files found in the directory")

    for file_name in file_names:
        file_path = os.path.join(dir, file_name)
        with open(file_path, "rb") as f:
            dict_load = json.load(f)
            pd_dict_list.append(dict_load)
    pd_avg_dict = {}
    for pd_dict in pd_dict_list:
        for i in pd_dict.keys():
            pd_avg_dict[int(i)] = pd_avg_dict.get(int(i)) + pd_dict[i] if (int(i) in pd_avg_dict.keys()) else pd_dict[i]
    for i in pd_avg_dict.keys():
        if not roundToInt:
            pd_avg_dict[i] = pd_avg_dict[i] / len(pd_dict_list)
        else:
            pd_avg_dict[i] = int(pd_avg_dict[i] / len(pd_dict_list))

    return pd_avg_dict
