import os
import argparse
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import re


def extract_fpr(file_path: str) -> float:
    """
    Use regex to extract the FPR value from the file path
    :param file_path: The path to the file
    :return: The FPR value as a float
    """
    pattern = r"/fpr_(\d+\.\d+)/fpr_\1\.txt"
    match = re.search(pattern, file_path)
    if match:
        fpr_value = float(match.group(1))
        return fpr_value
    else:
        raise ValueError(f"No FPR value found in the file path: {file_path}")


def parse_file(file_path: str, all_jaccard: Dict[float, Dict[str, List[Tuple[Tuple[str, str], float]]]]):
    """
    Parse the Jaccard similarity values from the file
    :param file_path: The path to the file
    :param all_jaccard: The dictionary to store the Jaccard similarity values
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    process_opt = None
    current_fpr = extract_fpr(file_path)
    parsing = False

    for line in lines:
        line = line.strip()
        if 'processed using' in line:
            process_opt = line.split()[-1].replace(']', '').lower()
            process_opt = "Coverage" if process_opt == "union" else "Stability"
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
                      stat: Dict[float, Dict[str, List[Tuple[Tuple[str, str], str]]]]):
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

            stat[fpr][process_opt] = []
            for pair, values in pair_values.items():
                result = f"{round(np.mean(values), 3):.3f} ± {round(np.std(values), 3):.3f}"
                stat[fpr][process_opt].append((pair, result))

def save_statistics(stat: Dict[float, Dict[str, List[Tuple[Tuple[str, str], str]]]], save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fpr, process_dict in stat.items():
        file_name = f"fpr_{fpr}/Jaccard_fpr_{fpr}.csv"
        full_path = os.path.join(save_dir, file_name)
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        with open(full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Process Option", "Pair 1", "Pair 2", "Value"])
            for process_opt, pairs in process_dict.items():
                for pair, value in pairs:
                    writer.writerow([process_opt, pair[0], pair[1], value])


def plot_similarity_matrix(dir_path: str, fpr_list: List[float], process_opt_list: List[str]):
    """
    Plot the similarity matrix
    :param dir_path: The directory path of the csv file and where to save the graphs
    :param fpr_list: The list of FPR values
    :param process_opt_list: The list of process options
    """
    for fpr in fpr_list:
        file_path = os.path.join(dir_path, f'fpr_{fpr}/Jaccard_fpr_{fpr}.csv')
        df = pd.read_csv(file_path)

        for process_option in process_opt_list:
            filtered_df = df[df['Process Option'] == process_option]

            attacks = sorted(set(filtered_df['Pair 1']).union(set(filtered_df['Pair 2'])))
            attacks = sorted(set(attack.split('_')[0] for attack in attacks))

            # Create an empty matrix
            matrix = pd.DataFrame(np.nan, index=attacks, columns=attacks)


            # Fill the matrix with the data
            for i, row in filtered_df.iterrows():
                p1, p2, value = row['Pair 1'], row['Pair 2'], row['Value']
                avg_value = float(value.split('±')[0].strip())
                short_p1 = p1.split('_')[0]
                short_p2 = p2.split('_')[0]
                matrix.loc[short_p1, short_p2] = avg_value
                matrix.loc[short_p2, short_p1] = avg_value

            # Plotting the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                matrix, annot=True, cmap='YlGnBu', linewidths=0.5, linecolor='gray', cbar=True, vmin=0, vmax=1,
                annot_kws={"size": 15, "weight": "bold"},  # Adjust annotation font size and weight
                fmt=".2f"
            )

            # Adjusting the font size and weight for labels
            plt.xticks(fontsize=12, weight='bold')
            plt.yticks(fontsize=12, weight='bold')

            # Reducing white space around the heatmap
            plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)

            # Save the plot
            save_path = os.path.join(dir_path, f'fpr_{fpr}/Jaccard_fpr_{fpr}_{process_option}.pdf')
            plt.savefig(save_path, bbox_inches='tight')  # Save with tight bounding box to reduce white space
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='obtain_mia_jaccard_similarity_matrix')
    parser.add_argument("--fpr", type=float, nargs="+", help="fpr list")
    parser.add_argument("--base_dir", type=str, help="base directory list")
    parser.add_argument("--plot_dir", type=str, help="plot directory")
    args = parser.parse_args()

    target_fpr = args.fpr
    all_jaccard = {fpr: {} for fpr in target_fpr}
    stat = {fpr: {} for fpr in target_fpr}

    base_dirs = args.base_dir.split()

    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if any(str(fpr) in file for fpr in target_fpr) and file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    if file_path and isinstance(file_path, str):
                        parse_file(file_path, all_jaccard)
                    else:
                        raise FileNotFoundError(f"File not found: {file_path}")

    calculate_avg_std(all_jaccard, stat)
    save_statistics(stat, args.plot_dir)
    plot_similarity_matrix(args.plot_dir, target_fpr, ['Coverage', 'Stability'])