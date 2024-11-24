# Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble

This directory contains the code for the experiments in the paper *Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble*. Note that throughout this repo, we refer coverage and stability (2 definition defined in the paper) as union and intersection respectively. We also refer instances and seeds, since each instance is `prepared` with a different seed.

## Table of Contents

- [Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble](#membership-inference-attacks-as-privacy-tools-reliability-disparity-and-ensemble)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Preparing Predictions of Multi-instances MIAs](#preparing-predictions-of-multi-instances-mias)
    - [`obtain_pred.py`](#obtain_predpy)
    - [`obtain_accuracy.py`](#obtain_accuracypy)
    - [`process_CINIC10.ipynb`](#process_cinic10ipynb)

## Abstract

> Membership inference attacks not only demonstrate a significant threat to the privacy of machine learning models but are also widely utilized as tools for privacy assessment, auditing, and machine unlearning. While prior research has focused primarily on developing new attacks with improved performance metrics such as AUC and TPR@low FPR, it often overlooks the disparities among different attacks and their reliability, which are crucial when MIAs are employed as privacy tools. This paper proposes a systematic evaluation of membership inference attacks from a novel perspective, highlighting significant issues of instability and disparity. We delve into the potential impacts and causes of these issues through extensive evaluations and discuss their implications. Furthermore, we introduce two ensemble strategies that harness the strengths of multiple existing attacks, and our experiment demonstrates their advantages in enhancing the effectiveness of membership inference.


To be able to set up the directory correctly, please replace the `DATA_DIR` in all the scripts with the path to the directory where you want to store the attack predictions and results.


-------------------
## Preparing Predictions of Multi-instances MIAs


### `obtain_pred.py`

1. Initialize the specified target model and the target dataset.
2. Split the target dataset into target dataset and auxiliary dataset.
3. Train the target model on the target dataset.
4. Prepare the specified MIAs with the (black box access) target model and the auxiliary dataset.
5. Save the prediction on the target dataset.

This file is being called by `experiment_script/prepare_target.sh` to prepare and save the target model and datasets. And it's also called by `experiment_scripts/obtain_pred.sh` to run the experiments.

Usage:

1. Prepare the target datasets and target models
    ```bash
    bash experiment_scripts/prepare_target.sh
    ```
2. Train attacks on the target models and target datasets, then save predictions
    ```bash
   bash experiment_scripts/obtain_pred.sh [seed]
   ```
   where the last number is the seed of that instance.

-------------------
## Comparing MIAs

### `obtain_graph.py`
The `obtain_graph.py` script is designed to load data, generate various plots, and evaluate metrics. 
The code is divided into three primary categories: Data Loading, Plot Diagram, and Evaluation.

- Data Loading

   - **`load_and_create_predictions`**: Loads data and create prediction object used in later evaluations and plotting.
   - **`load_diff_distribution`**: Loads data and create prediction object for *same attack different distribution* case.

- Plot Diagram
   
   - **`plot_venn`**: Plots a Venn diagram for comparisons between attackss.
      ```bash
      bash experiment_scripts/obtain_venn.sh 
       ``` 
   - **`plot_auc`**: Plots a AUC diagram (Area Under the Curve)  for different models or attacks.
      ```bash
     bash experiment_scripts/obtain_auc.sh
       ```
   - **`multi_seed_convergence`**: Visualizes the convergence across multiple seeds for the model or experiment.   
      ```bash
     bash experiment_scripts/obtain_multi_seed_conv.sh
       ```


  
### `obtain_jaccard.py`
   The `obtain_jaccard.py` is designed to save the Jaccard similarity between different MIAs and plot a heatmap to visualize the Jaccard similarity matrix.
   ```bash
   bash experiment_scripts/obtain_jaccard.sh

   ```

### `obtain_accuracy.py`
   The `obtain_accuracy.py` is designed to save the accuracy of different MIAs under TPR@FPRs and balanced accuracy.
   ```bash
    bash experiment_scripts/obtain_accuracy.sh
   ```
   
### `process_CINIC10.ipynb`
   The `process_CINIC10.ipynb` is designed to process the CINIC10 dataset 30,000 ImageNet samples and 30,000 CIFAR10 samples._