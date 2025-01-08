# Membership Inference Attacks (MIA)

This directory contains the implementation of various Membership Inference Attacks (MIA). All attacks are based on the classes defined in the `base.py` file.

## Base Classes

### ModelAccessType

This is an Enum class that defines the type of access to the model. It can be one of the following:

- WHITE_BOX: Full access to the model, meaning the entire model checkpoint is available.
- BLACK_BOX: Only access to the model's outputs given its inputs, and also the model structure.
- GRAY_BOX: Partial access to the model, more than black box but less than white box (Most attacks wouldn't be using this Access).

### AuxiliaryInfo

This is an abstract base class for all auxiliary information. The auxiliary information is used to assist in the attack and specify the attack (ie. training epochs, optimizer, etc.). The specific type and use of the auxiliary information will depend on the specific attack.

### ModelAccess

This is an abstract base class for all types of model access. It defines the interface for accessing the model. The specific type of access (white box, black box, or gray box) is specified when an instance of a subclass is created.

### MiAttack

This is an abstract base class for all attacks. It defines the interface for an attack. An attack is initialized with a model access, auxiliary information, and optionally target data. The attack can then be prepared and used to infer the membership of data.

## Usage

To implement a new attack, create a new file in this directory. In this file, define a new class that inherits from `MiAttack`. Implement the required methods `prepare` and `infer`. You may also need to define new subclasses of `ModelAccess` and `AuxiliaryInfo` if your attack requires a new type of model access or auxiliary information.

For examples of how to implement an attack, see the existing attack implementations in this directory.

## Attacks
| filename             | Attack                  | Paper                                                                                  |
|----------------------|-------------------------|----------------------------------------------------------------------------------------|
| `aug_mia.py`         | Augmentation Attack     | Label-Only Membership Inference Attacks, PMLR 2021                                     |
| `calibration_mia.py` | Calibration Loss Attack | On the Importance of Difficulty Calibration in Membership Inference Attacks, ICLR 2022 |
| `lira_mia`           | LiRA Attack             | Membership Inference Attacks From First Principles", S&P 2022                          |
| `shokri_mia.py`      | Class-NN Attack         | Membership Inference Attacks Against Machine Learning Models, IEEE S&P 2017            |
| `losstraj_mia.py`    | Loss Trajectory Attack  | Membership Inference Attacks by Exploiting Loss Trajectory, CCS 2022                   |
| `reference_mia.py`   | Reference Attack        | Enhanced Membership Inference Attacks against Machine Learning Models, CCS 2022        |
| `yeom_mia.py`        | LOSS Attack             | Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting, IEEE S&P 2018|

Any attack file not listed here are not fully implemented yet.