# Venn Diagram Analysis

This file focuses on the generation and analysis of Venn diagrams for comparing different sets. 
The code is organized into three main categories: Similarity Metrics, Data Processing, and Plotting Venn Diagrams.

## Similarity Metrics

This section includes functions to calculate similarity between sets using different metrics. The available metrics are:

- **`jaccard_similarity`**: Computes the Jaccard similarity between two sets.
- **`overlap_coefficient`**: Measures the overlap between two sets.
- **`set_size_variance`**: Calculates the variance in sizes of multiple sets.
- **`entropy`**: Computes the entropy based on set distributions.

## Data Processing

Data processing is essential for organizing and preparing the data to be visualized in Venn diagrams. The key processing functions include:

- **`data_process_for_venn`**: Prepares data for Venn diagram input.
- **`single_attack_process_for_venn`**: Processes data for *a single attack with several different seeds* case.
- **`single_seed_process_for_venn`**: Processes data for *several attacks with only a single seed* case.

## Plotting Venn Diagrams

This section contains functions for plotting Venn diagrams based on processed data:

- **`plot_venn_single`**: Plots a Venn diagram for a single attack for at most 3 seeds.
- **`plot_venn_single_for_all_seed`**: Plots Venn diagrams for at most 6 seeds.
- **`plot_venn_pairwise`**: Plots pairwise Venn diagrams.
- **`plot_venn_diagram`**: General function for plotting Venn diagrams for at most 3 seeds.
- **`plot_venn_for_all_attacks`**: Plots Venn diagrams for all attacks for at most 6 seeds.
