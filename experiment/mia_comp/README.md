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