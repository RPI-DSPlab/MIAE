import os
from typing import List, Dict, Tuple
import numpy as np
import csv

def parse_file(file_path: str, all_jaccard: Dict[float, Dict[str, List[Tuple[Tuple[str, str], float]]]]):
    """
    Parse the Jaccard similarity values from the file
    :param file_path: The path to the file
    :param all_jaccard: The dictionary to store the Jaccard similarity values
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    process_opt = None
    current_fpr = None
    parsing = False

    for line in lines:
        line = line.strip()
        if 'processed using' in line:
            parts = line.split(',')
            for part in parts:
                if 'FPR =' in part:
                    current_fpr = float(part.split('=')[-1].strip().split()[0])
            process_opt = line.split()[-1].replace(']', '').lower()
            if current_fpr in all_jaccard and process_opt not in all_jaccard[current_fpr]:
                all_jaccard[current_fpr][process_opt] = []

        elif '(1) Pairwise Jaccard Similarity' in line:
            parsing = True
        elif '(2) Average Jaccard Similarity' in line:
            parsing = False

        if parsing and current_fpr in all_jaccard and 'vs' in line:
            parts = line.split(':')
            pair = tuple(parts[0].strip().split(' vs '))
            value = float(parts[1].strip())
            all_jaccard[current_fpr][process_opt].append((pair, value))

def calculate_avg_std(all_jaccard: Dict[float, Dict[str, List[Tuple[Tuple[str, str], float]]]],
                      stat: Dict[float, Dict[str, List[Tuple[Tuple[str, str], float]]]]):
    """
    Calculate the average and standard deviation of the Jaccard similarity values
    :param all_jaccard: The dictionary containing the Jaccard similarity values
    :param stat: The dictionary to store the average and standard deviation of the Jaccard similarity values
    """
    for fpr, process_dict in all_jaccard.items():
        for process_opt, jaccard_list in process_dict.items():
            pair_values = {}
            for pair, value in jaccard_list:
                if pair not in pair_values:
                    pair_values[pair] = []
                pair_values[pair].append(value)

            # for pair, values in pair_values.items():
            #     print(f"FPR: {fpr}, values: {values}, Pair: {pair}, Avg: {np.mean(values):.4f}, Std Dev: {np.std(values):.5f}")

            stat[fpr][process_opt] = []
            for pair, values in pair_values.items():
                result = f"{round(np.mean(values), 3):.3f} ± {round(np.std(values), 3):.3f}"
                stat[fpr][process_opt].append((pair, result))

def save_statistics(stat: Dict[float, Dict[str, List[Tuple[Tuple[str, str], str]]]], save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fpr, process_dict in stat.items():
        file_name = f"Jaccard_fpr_{fpr}.csv"
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Process Option", "Pair 1", "Pair 2", "Value"])
            for process_opt, pairs in process_dict.items():
                for pair, value in pairs:
                    writer.writerow([process_opt, pair[0], pair[1], value])

if __name__ == "__main__":
    target_fpr = [0.1, 0.3]
    all_jaccard = {fpr: {} for fpr in target_fpr}
    stat = {fpr: {} for fpr in target_fpr}

    base_dirs = [
        "/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_0/graphs/venn/fpr/common_tp",
        "/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_1/graphs/venn/fpr/common_tp",
        "/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_2/graphs/venn/fpr/common_tp",
        "/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_3/graphs/venn/fpr/common_tp"
    ]

    for base_dir in base_dirs:
        print(f"Processing {base_dir}")
        for root, _, files in os.walk(base_dir):
            for file in files:
                if any(str(fpr) in file for fpr in target_fpr) and file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    if file_path:
                        parse_file(file_path, all_jaccard)
                    else:
                        raise FileNotFoundError(f"File not found: {file_path}")

    # Print all jaccard results
    # for fpr, jaccard_dict in all_jaccard.items():
    #     print(f"FPR: {fpr}")
    #     for process_opt, jaccard_list in jaccard_dict.items():
    #         print(f"Process Option: {process_opt}")
    #         for pair, value in jaccard_list:
    #             print(f"{pair}: {value}")
    #         print()
    #     print()

    calculate_avg_std(all_jaccard, stat)
    save_dir = "/data/public/comp_mia_data/repeat_exp_set/eval_stat"
    save_statistics(stat, save_dir)

    # Print the average and standard deviation of the Jaccard similarity values
    for fpr, process_dict in stat.items():
        print(f"FPR: {fpr}")
        for process_opt, jaccard_list in process_dict.items():
            print(f"Process Option: {process_opt}")
            for pair, result in jaccard_list:
                print(f"{pair}: {result}")
            print()
        print()
