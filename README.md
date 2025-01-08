# MIAE (Membership Inference Attacks and Evaluation)

The **Membership Inference Attacks Evaluation (MIAE)** package is a Python library designed to facilitate the evaluation of membership inference attacks on machine learning models. It offers a comprehensive framework for implementing, testing, and comparing various types of membership inference attacks.

To obtain a set of predictions from an attack, we follow these three steps:

1.	**Instantiate the attack** with its AuxiliaryInfo (configuration) and target model access.

2.	**Prepare the attack** by calling .prepare() with the auxiliary dataset.

3.	**Perform inference** by calling .infer() on the target dataset.

When assessing multiple attacks, we often provide the same target model access, auxiliary dataset, and target dataset as API inputs to all attacks to ensure consistent experimental conditions. The membership inference scores are also represented as objects, facilitating easier evaluation and comparison.

We have implemented seven membership inference attacks within MIAE, ranging from basic methods like the **Loss Threshold Attack** by Yeom et al. and the **Class-NN Attack** by Shokri et al., to more advanced techniques like **LiRA** by Carlini et al. Additionally, the package integrates evaluation methods directly into the membership prediction objects, including metrics like True Positive Rate at low False Positive Rate (TPR@low FPR), similarity measures, and visualization tools like Venn diagrams.

`/miae` subdirectory contains the MIAE package.


`/experiment` subdirectory contains projects using MIAE package. They also serve as examples of using MIAE package.


---
## Set up the environment
```bash
conda env create -f miae_env.yml
conda activate miae
```

## Citation (To be implemented)
If you use MIAE in your research, please cite our paper:
```
@inproceedings{


}
```