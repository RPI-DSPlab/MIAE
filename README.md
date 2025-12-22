# MIAE (Membership Inference Attacks and Evaluation) 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15491989.svg)](https://doi.org/10.5281/zenodo.15491989) [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2506.13972)

MIAE is a versatile Python framework for evaluating Membership Inference Attacks (MIAs), emphasizing the disparity of MIAs. It provides a modular interface for implementing and evaluating various MIA strategies against machine learning models.

Throughout this repo, we refer to coverage and stability (two definitions introduced in the paper) as union and intersection respectively. We also refer to instances and seeds, since each instance is `prepared` with a different seed.

 ## Table of Contents

- [MIAE (Membership Inference Attacks and Evaluation)](#miae-membership-inference-attacks-and-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. Using as a Python Library](#1-using-as-a-python-library)
    - [2. Experimentation \& Reproducing Results](#2-experimentation--reproducing-results)
  - [Citation](#citation)
 



 ## Installation

 You can install the package and its dependencies using conda and pip.

 ```bash
 # 1. Create the environment
 conda env create -f miae_env.yml
 conda activate miae

 # 2. Install the package in editable mode
 pip install -e .
 ```
 
 ## Usage

 There are two primary ways to use this repository:

 1. **As a Python Library**: Import `miae` to run membership inference attacks on your own models and datasets.
 2. **For Experimentation**: Use and modify the scripts in the `experiment/` directory to test hypotheses using our existing experimental pipeline.

 ### 1. Using as a Python Library

 Here is an example using the Yeom et al. attack:

 ```python
 import torch
 from miae.attacks import yeom_mia

 # 1. Setup your model and data
 # target_model = ... (Your trained PyTorch model)
 # untrained_model = ... (A fresh instance of your model architecture)
 # aux_dataset = ... (Dataset for attack preparation/shadow training)
 # target_dataset = ... (Dataset to attack)

 # 2. Define Model Access
 # Wraps the target model to provide the specific access level required by the attack
 model_access = yeom_mia.YeomModelAccess(target_model, untrained_model)

 # 3. Configure Attack Parameters
 config = {
     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
     'num_classes': 10,
     'batch_size': 128,
     'epochs': 10,
     'lr': 0.1,
     'save_path': './temp',  # For saving intermediate files
     'log_path': './logs'
 }
 aux_info = yeom_mia.YeomAuxiliaryInfo(config)

 # 4. Initialize and Run Attack
 attack = yeom_mia.YeomAttack(model_access, aux_info)

 # Prepare the attack (train shadow models, calculate stats, etc.)
 attack.prepare(aux_dataset)

 # Run inference on the target dataset
 # Returns membership scores/probabilities
 predictions = attack.infer(target_dataset)
 ```

 For more details on the attacks implemented in this repo, please refer to [attacks](miae/attacks/README.md).

 ### 2. Experimentation & Reproducing Results

 To reproduce the results from our paper or run your own experiments,  check [mia_comp](experiment/mia_comp/README.md). Each sub-directory contains scripts and notebooks for specific experiments.

 ---

 ## Citation

 If you use this package in your research, please cite our paper:

 ```bibtex
@inproceedings{10.1145/3719027.3744818,
author = {Wang, Zhiqi and Zhang, Chengyu and Chen, Yuetian and Baracaldo, Nathalie and Kadhe, Swanand R and Yu, Lei},
title = {Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble},
year = {2025},
isbn = {9798400715259},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3719027.3744818},
doi = {10.1145/3719027.3744818},
booktitle = {Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},
series = {CCS '25}
}
 ```

