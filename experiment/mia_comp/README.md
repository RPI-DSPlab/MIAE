# mia_comp (exact name TBD)

This is the directory dedicated for finding the common attacked datapoints across different Membership Inference Attacks (MIA).

## Project Structure

The main script of the project is `obtain-roc.py` and `analysis_main.py`, which performs the following steps:

### `obtain-roc.py`

1. Initialize the specified target model and the target dataset.
2. Split the target dataset into target dataset and auxiliary dataset.
3. Train the target model on the target dataset.
4. Prepare the specified MIA with the (black box access) target model and the auxiliary dataset.
5. Save the prediction on the target dataset.

### `analysis_main.py`
1. Load the predictions on the same target dataset from multiple MIAs.
2. Perform the analysis on the predictions across different MIAs.


## Usage

1. Prepare the target datasets and target models
    ```bash
    bash experiment_scripts/prepare_target.sh
    ```
2. Train attacks on the target models and target datasets, then save predictions
    ```bash
   bash experiment_scripts/obtain_pred.sh 0
   ```
   where the last number is the seed.

The python environment needed is provided at `miae-conda-env.yaml`. To create the environment, run the following command:

```bash 
conda env create -f miae-conda-env.yaml
```

-------------------
# obtain_graph.py

The `obtain_graph.py` script is designed to load data, generate various plots, and evaluate metrics. 
The code is divided into three primary categories: Data Loading, Plot Diagram, and Evaluation.

## Data Loading

This section handles loading and processing data required for plotting and analysis. The key functions include:

- **`load_and_create_predictions`**: Loads data and create prediction object used in later evaluations and plotting.
- **`load_diff_distribution`**: Loads data and create prediction object for *same attack different distribution* case.

## Plot Diagram

This section focuses on generating different types of plots for analysis, including Venn diagrams, AUC curves, and distribution plots:

- **`plot_venn`**: Plots a Venn diagram for comparisons between attackss.
- **`plot_auc`**: Plots a AUC diagram (Area Under the Curve)  for different models or attacks.
- **`plot_hardness_distribution`**: Plots the distribution of data hardness, useful for understanding which data points are more difficult to classify.
- **`plot_hardness_distribution_unique`**: Plots a unique distribution of data hardness.
- **`multi_seed_convergence`**: Visualizes the convergence across multiple seeds for the model or experiment.

## Evaluation

This section includes evaluation functions for analyzing results from different experiments and attacks:

- **`single_attack_seed_ensemble`**: Evaluates the performance of a single attack across multiple seeds using ensemble techniques.
- **`eval_metrics`**: Computes and evaluates various similarity metrics.
