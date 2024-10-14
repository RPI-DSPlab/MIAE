"""
This file consist classes and functions that are used to ensemble
attacks, and scripts to read attack results and ensemble them.
We may later move the ensemble functions and classes to MIAE
framework.

what does this file do?
1. train shadow-target model based on pre-saved auxiliary set
2. prepare ensemble: train the ensemble on the prediction on shadow-target model
3. ensemble the attack results
"""

import argparse
import os
import pickle
import re
import sys
from tqdm import tqdm
from torchvision.models import resnet18

sys.path.append(os.path.join(os.getcwd(), "..", ".."))

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from typing import List, Dict, Tuple
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
import miae.eval_methods.prediction as prediction
from miae.utils.set_seed import set_seed
from miae.utils import dataset_utils
from experiment import models
from experiment.models import get_model
from obtain_pred import train_target_model
from miae.eval_methods.experiment import load_target_dataset, read_preds


# --------------------- functions and classes for ensemble -------------------------


class StackingMetaModel(nn.Module):
    """MetaModel for stacking"""

    def __init__(self, input_dim: int, logits_length):
        super(StackingMetaModel, self).__init__()
        self.fc1 = nn.Linear(logits_length + input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)  # Adding dropout for regularization

    def forward(self, x, logits):
        x = torch.cat((x, logits), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CustomEncoder(nn.Module):
    def __init__(self):
        super(CustomEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 16 x 16

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64 x 8 x 8

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128 x 4 x 4

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256 x 2 x 2

        # Fully connected layer
        self.fc = nn.Linear(256 * 2 * 2, 512)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc(x))
        return x


class LearningBasedMetaModel(nn.Module):
    def __init__(self, cv_model: nn.Module, num_attack_predictions, fine_tune=True):
        super(LearningBasedMetaModel, self).__init__()
        self.cv_model = cv_model

        if not fine_tune:
            for param in self.cv_model.parameters():
                param.requires_grad = False

        self.cv_model.fc = nn.Identity()

        # Fully connected layers for combining image features and attack predictions
        self.fc1 = nn.Linear(1024 + num_attack_predictions, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 2)  # output_dim=2: membership prediction is binary

    def forward(self, image, attack_predictions):
        image_features = self.cv_model(image)

        # Concatenate image features with attack predictions
        combined_features = torch.cat((image_features, attack_predictions), dim=1)

        # Fully connected layers
        x = torch.relu(self.fc1(combined_features))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def mia_ensemble_stacking_prepare(base_preds: List[prediction.Predictions], gt: np.array, logits, device="cuda:0"):
    """
    This function ensembles the predictions of the base models using stacking

    :param base_preds: List of predictions of the base models
    :param gt: Ground truth labels
    """

    num_base_model = len(base_preds)
    meta_model = StackingMetaModel(input_dim=num_base_model, logits_length=len(logits[0]))
    # Ensure the numpy arrays are converted to torch.float32 tensors
    meta_features = torch.stack([torch.from_numpy(p.pred_arr).float() for p in base_preds],
                                dim=1)  # stack all predictions from attacks
    meta_labels = torch.tensor(gt, dtype=torch.long)

    # move everything to device
    meta_model.to(device)
    meta_labels = meta_labels.to(device)
    meta_features = meta_features.to(device)
    logits = torch.tensor(logits, dtype=torch.float32).to(device)

    # Train the meta model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = meta_model(meta_features, logits)
        loss = criterion(outputs, meta_labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total = meta_labels.size(0)
            correct = (predicted == meta_labels).sum().item()
            print(f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {100 * correct / total}")

    return meta_model


def mia_ensemble_avg(base_preds: List[prediction]):
    """
    This function ensembles the predictions of the base models using averaging

    :param base_preds: List of predictions of the base models
    """

    num_base_model = len(base_preds)
    num_samples = base_preds[0].pred_arr.shape[0]
    ensemble_preds = np.zeros_like(base_preds[0].pred_arr, dtype=np.float32)
    for p in base_preds:
        ensemble_preds += p.pred_arr.astype(np.float32)
    ensemble_preds /= num_base_model

    return ensemble_preds


def mia_learning_based_ensemble_prepare(aux_set, base_preds: List[prediction.Predictions], gt: np.array,
                                        cv_model_choice: str = "custom_encoder", unfreeze_epoch=5, device="cuda:0"):
    """
    This function prepare to ensemble the predictions of the base models using a learning-based approach

    :param aux_set: auxiliary set used to train the learning-based meta model
    :param base_preds: List of predictions of the base models
    :param gt: Ground truth labels
    :param cv_model_choice: choice of the convolutional model to use for the learning-based meta model [pretrained_resnet18]
    """

    # Prepare the features for the meta model
    num_attack_predictions = len(base_preds)
    base_preds_stacked = torch.stack([torch.from_numpy(p.pred_arr).float() for p in base_preds], dim=1)
    meta_labels = torch.tensor(gt, dtype=torch.long)

    # Prepare the image features for the meta model
    images = []
    for data in aux_set:
        images.append(data[0])
    images = torch.stack(images)

    # Train the learning-based meta model
    if cv_model_choice == "pretrained_resnet18":
        cv_model = resnet18(pretrained=True)
    elif cv_model_choice == "custom_encoder":
        cv_model = CustomEncoder()
    meta_model = LearningBasedMetaModel(cv_model, num_attack_predictions, fine_tune=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)

    # move everything to device
    meta_model.to(device)
    meta_labels = meta_labels.to(device)
    base_preds_stacked = base_preds_stacked.to(device)
    images = images.to(device)
    for epoch in tqdm(range(60)):
        optimizer.zero_grad()
        outputs = meta_model(images, base_preds_stacked)
        loss = criterion(outputs, meta_labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total = meta_labels.size(0)
            correct = (predicted == meta_labels).sum().item()
            print(f"Epoch {epoch}, Loss: {loss.item():.2f}, Accuracy: {100 * correct / total}")

    return meta_model


def obtain_shadow_roc(preds_on_shadow_target: List[prediction.Predictions], gt: np.array) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    for each attack, calculate the FPR thresholds and their corresponding FPRs and TPRs

    preds_on_shadow_target: list of predictions on shadow-target model
    gt: ground truth labels

    """
    ret_list = []
    for pred in preds_on_shadow_target:
        ret_list.append(prediction.roc_curve(gt, pred.pred_arr))
    return ret_list



def align_threshold_by_fpr(fpr_thresholds: List[Tuple[np.ndarray, np.ndarray]],
                           target_fprs: list[float]) -> List[Dict[float, float]]:
    """
    Process fpr_thresholds pairs for multiple attacks. For each attack, find fpr that's closest to each target_fpr and
    save the corresponding threshold. Hence we are aligning the thresholds for different attack that would lead to the
    same fpr.

    fpr_thresholds: list of fpr-threshold pair. One pair for each attack.
    target_fprs: list of fpr values to align to.

    return: list of dictionary. Each dictionary contains the threshold for that attack at each target fpr.
    """


    ret_list = []
    for fpr_arr, threshold_arr in fpr_thresholds: # for each attack
        thresholds = [] # threshold for each attack at the target fpr
        for target_fpr in target_fprs:
            if len(fpr_arr[fpr_arr <= target_fpr]) == 0:
                thresholds.append(np.inf)  # no threshold can achieve the target fpr
            else:
                idx = np.argmin(abs(fpr_arr - target_fpr))
                thresholds.append(threshold_arr[idx])

        ret_list.append(dict(zip(target_fprs, thresholds)))
    return ret_list
    


def pairwise_max(base_preds: List[prediction.Predictions]) -> dict:
    """
    This function ensembles every 2 predictions by taking the maximum of the two predictions

    :param base_preds: List of predictions of the base models

    :return: Dictionary of each pair name as the key and their ensemble predictions as the value
    """

    num_base_model = len(base_preds)
    ensemble_preds = {}
    for i in range(num_base_model):
        for j in range(i + 1, num_base_model):
            name_i = base_preds[i].name.split('_')[-1]
            name_j = base_preds[j].name.split('_')[-1]
            key = f"{name_i}_{name_j}_max"
            ensemble_preds[key] = np.maximum(base_preds[i].pred_arr, base_preds[j].pred_arr)

    return ensemble_preds


# --------------------- helping functions -----------------------------------------



def obtain_logits(model, data, device):
    """
    obtain_logits obtain the logits of the model

    model: model to obtain the logits
    data: data to obtain the logits
    device: device to run the model
    """

    dataloader = DataLoader(data, batch_size=512, shuffle=False, num_workers=2)
    model.eval()
    model.to(device)
    logits = []
    with torch.inference_mode():
        for images, _ in dataloader:
            images = images.to(device)
            logits.append(model(images).detach().cpu().numpy())
    return np.concatenate(logits)


def prepare_ensemble(preds_on_shadow_target: List[prediction.Predictions], gt, ensemble_method: str, save_path: str,
                     aux_set, shadow_target_logits, device):
    """
    prepare_ensemble prepares a ensemble and save the ensemble model if needed

    preds_on_shadow_target: list of predictions on shadow-target model
    gt: ground truth labels
    ensemble_method: method to ensemble the predictions [avg, stacking]
    save_path: path to save the ensemble model
    """
    if ensemble_method == "avg":
        print("NO need to prepare for avg")
    elif ensemble_method == "stacking":
        meta_model = mia_ensemble_stacking_prepare(preds_on_shadow_target, gt, shadow_target_logits, device)
        torch.save(meta_model, save_path + "/stacking_meta_model.pth")
    elif ensemble_method == "learning_based_cnn_pretrained":
        meta_model = mia_learning_based_ensemble_prepare(aux_set, preds_on_shadow_target, gt, device=device)
        torch.save(meta_model, save_path + "/learning_based_meta_model.pth")
    elif ensemble_method == "pairwise_max":
        rocs = obtain_shadow_roc(preds_on_shadow_target, gt)
        threshold_fpr_list = [tuple([x[0], x[2]]) for x in rocs]
        shadow_attack_thresholds = align_threshold_by_fpr(threshold_fpr_list, [0.01, 0.05, 0.1])
        with open(save_path + "/shadow_attack_threshold_fpr.pkl", "wb") as f:
            pickle.dump(shadow_attack_thresholds, f)
        print(f"shadow_attack_thresholds is saved at {save_path}/shadow_attack_threshold_fpr.pkl")
    else:
        raise ValueError("Invalid ensemble method")


def run_ensemble(base_preds: List[prediction.Predictions], dataset_to_attack: Dataset, ensemble_method: str,
                 result_save_path: str, target_logits, ensemble_save_path: str, device):
    """
    run the ensemble on the dataset_to_attack

    base_preds: list of predictions to be ensemble
    dataset_to_attack: dataset to attack
    ensemble_method: method to ensemble the predictions [avg, stacking]
    save_path: path to save the ensemble result
    target_logits: target_logits of the target model
    ensemble_save_path: path to saved ensemble file (ie: meta model)
    """

    if ensemble_method == "avg":
        ensemble_preds = mia_ensemble_avg(base_preds)
    elif ensemble_method == "stacking":
        meta_model = torch.load(ensemble_save_path + "/stacking_meta_model.pth")
        meta_model.to(device)
        meta_features = torch.stack([torch.from_numpy(p.pred_arr).float() for p in base_preds], dim=1)
        target_logits = torch.tensor(target_logits, dtype=torch.float32)
        ensemble_preds = meta_model(meta_features.to(device), target_logits.to(device)).cpu().detach().numpy()
    elif ensemble_method == "learning_based_cnn_pretrained":
        meta_model = torch.load(ensemble_save_path + "/learning_based_meta_model.pth")
        base_preds_stacked = torch.stack([torch.from_numpy(p.pred_arr).float() for p in base_preds], dim=1).to(device)
        images = []
        meta_model.eval()
        meta_model.to(device)
        for data in dataset_to_attack:
            images.append(data[0])
        images = torch.stack(images).to(device)
        ensemble_preds = meta_model(images, base_preds_stacked).cpu().detach().numpy()
        # retrieve the membership prediction
        ensemble_preds = ensemble_preds[:, 1]
        # normalize the prediction
        ensemble_preds = (ensemble_preds - np.min(ensemble_preds)) / (np.max(ensemble_preds) - np.min(ensemble_preds))
    elif ensemble_method == "pairwise_max":
        ensemble_preds = pairwise_max(base_preds)
        for name, pred in ensemble_preds.items():
            with open(result_save_path + f"/ensemble_preds_{name}.pkl", "wb") as f:
                pickle.dump(pred, f)
        return  # this methods saves in a different way
    else:
        raise ValueError("Invalid ensemble method")

    # save the ensemble result
    with open(result_save_path + f"/ensemble_preds_{ensemble_method}.pkl", "wb") as f:
        if ensemble_method == "stacking":
            pickle.dump(np.transpose(ensemble_preds)[1], f)
        else:
            pickle.dump(ensemble_preds, f)


def get_ensemble_methods(directory):
    files = os.listdir(directory)
    methods = []
    fns = []

    # Regex pattern to match the ensemble method names
    pattern = re.compile(r'ensemble_preds_(.+)\.pkl')

    for file in files:
        # Match the pattern to extract the method name
        match = pattern.match(file)
        if match:
            methods.append(match.group(1))
            fns.append(file)

    return methods, fns


def multi_seed_avg(pred_list: List[List[prediction.Predictions]]) -> List[prediction.Predictions]:
    """
    Average the predictions across multiple seeds

    :param pred_list: List of predictions from multiple seeds that follows the structure of read_preds
    :return: List of averaged predictions
    """

    avg_preds = []
    pred_names = [p.name for p in pred_list[0]]
    for i in range(len(pred_list[0])):
        pred_attack_i = []
        for j in range(len(pred_list)):
            pred_attack_i.append(pred_list[j][i])
        avg_preds.append(prediction.multi_seed_ensemble(pred_attack_i, "avg"))
        avg_preds[-1].name = pred_names[i]

    return avg_preds



# --------------------- scripts for each mode of this file  -------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble the attack results')
    parser.add_argument('--mode', type=str, help='[train_shadow, train_ensemble, run_ensemble]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to train the model')
    # ----- train_shadow arguments -----
    parser.add_argument('--shadow_model', type=str, help='Shadow model to train')
    parser.add_argument('--aux_set_path', type=str, help='auxiliary set path')
    parser.add_argument('--shadow_save_path', type=str, help='Save path for shadow model')
    parser.add_argument('--target_model', type=str, default=None,
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument('--data_aug', type=bool, default=False, help='whether to use data augmentation')
    parser.add_argument('--attack_lr', type=float, default=0.1, help='learning rate for MIA training')
    parser.add_argument('--attack_epochs', type=int, default=100, help='number of epochs for MIA training')
    parser.add_argument('--target_epochs', type=int, default=100, help='number of epochs for target model training')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset: [cifar10, cifar100]')
    # ----- prepare ensemble arguments ----
    parser.add_argument('--preds_path', type=str,
                        help='Save path for the predictions (both shadow preds and base preds)')
    parser.add_argument("--ensemble_seeds", type=int, nargs="+", help="Random seed")
    parser.add_argument("--attacks", type=str, nargs="+", default=None,
                        help='MIA type: [losstraj, yeom, shokri, aug, lira, calibration, reference]')
    parser.add_argument("--ensemble_save_path", type=str, help="Path to save the ensemble files (ie: meta model)")
    parser.add_argument('--shadow_target_data_path', type=str,
                        help='Save path for shadow target data, ground truth, etc')
    parser.add_argument('--ensemble_method', type=str,
                        help='Ensemble method [avg, stacking, learning_based_cnn_pretrained, pairwise_max]')
    # ---- run ensemble arguments ----
    parser.add_argument('--target_data_path', type=str, help='path to target data')
    parser.add_argument('--ensemble_result_path', type=str, help='path to save the ensemble pred result')
    parser.add_argument('--target_model_path', type=str, help='path to target model')
    args = parser.parse_args()

    set_seed(args.seed)
    if args.mode == "train_shadow":  # train shadow-target model
        with open(os.path.join(args.aux_set_path, "aux_set.pkl"), "rb") as f:
            original_aux_set = pickle.load(f)

        # re-partition the auxiliary set
        target_set_len = int(len(original_aux_set) / 2)
        target_set, aux_set = dataset_utils.dataset_split(original_aux_set,
                                                          [target_set_len, len(original_aux_set) - target_set_len])
        target_train_set, target_test_set = dataset_utils.dataset_split(target_set, [int(0.5 * target_set_len),
                                                                                     int(0.5 * target_set_len)])

        dataset_to_attack = ConcatDataset([target_train_set, target_test_set])
        target_membership = np.concatenate([np.ones(len(target_train_set)), np.zeros(len(target_test_set))])

        # training the target model
        if args.dataset == "cifar10":
            num_classes = 10
            input_size = 32
        elif args.dataset == "cifar100":
            num_classes = 100
            input_size = 32
        if args.dataset == "cinic10":
            num_classes = 10
            input_size = 32
        else:
            raise ValueError("Invalid dataset")
        target_model = models.get_model(args.target_model, num_classes, input_size).to(args.device)
        target_model = train_target_model(target_model, args.shadow_save_path, args.device, target_train_set,
                                          target_test_set, args)

        # save datasets
        dataset_save_path = os.path.join(args.shadow_save_path, f"{args.dataset}")
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)
        with open(os.path.join(dataset_save_path, "target_trainset.pkl"), "wb") as f:
            pickle.dump(target_train_set, f)
        with open(os.path.join(dataset_save_path, "target_testset.pkl"), "wb") as f:
            pickle.dump(target_test_set, f)
        with open(os.path.join(dataset_save_path, "aux_set.pkl"), "wb") as f:
            pickle.dump(aux_set, f)
        index_to_data = {}
        for i in range(len(dataset_to_attack)):
            index_to_data[i] = dataset_to_attack[i]

        with open(os.path.join(dataset_save_path, "index_to_data.pkl"), "wb") as f:
            pickle.dump(index_to_data, f)

        # save the membership
        np.save(os.path.join(dataset_save_path, "attack_set_membership.npy"), target_membership)

    elif args.mode == "train_ensemble":
        # read the predictions
        pred_path = args.preds_path

        # read data
        with open(os.path.join(args.shadow_target_data_path, "target_trainset.pkl"), "rb") as f:
            target_trainset = pickle.load(f)
        with open(os.path.join(args.shadow_target_data_path, "target_testset.pkl"), "rb") as f:
            target_testset = pickle.load(f)
        with open(os.path.join(args.shadow_target_data_path, "aux_set.pkl"), "rb") as f:
            aux_set = pickle.load(f)
        index_to_data, membership = load_target_dataset(args.shadow_target_data_path)

        shadow_target_preds = read_preds(pred_path, "_ensemble_base", args.ensemble_seeds, args.dataset,
                                         args.target_model,
                                         args.attacks, membership)

        # prepare the ensemble (single seed)
        preds_on_shadow_target_list = shadow_target_preds[0]  # getting preds from the 0th seeds' prediction
        save_path = os.path.join(args.ensemble_save_path, "single_seed")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shadow_target_model = get_model(args.target_model, 10, 32)
        shadow_target_model.load_state_dict(
            torch.load(os.path.join(args.shadow_save_path, "target_model_resnet56cifar10.pkl")))
        shadow_target_model.to(args.device)
        # prepare shadow target logits
        dataset_to_attack = ConcatDataset([target_trainset, target_testset])
        shadow_target_logits = obtain_logits(shadow_target_model, dataset_to_attack, args.device)
        prepare_ensemble(preds_on_shadow_target_list, membership, args.ensemble_method, save_path, aux_set,
                         shadow_target_logits, args.device)

        # prepare the ensemble (multi-seed)
        preds_on_shadow_target_list_multiseed_avg = multi_seed_avg(shadow_target_preds)
        save_path = os.path.join(args.ensemble_save_path, "multi_seed")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dataset_to_attack = ConcatDataset([target_trainset, target_testset])
        shadow_target_logits = obtain_logits(shadow_target_model, dataset_to_attack, args.device)
        shadow_target_model = get_model(args.target_model, 10, 32)
        shadow_target_model.load_state_dict(
            torch.load(os.path.join(args.shadow_save_path, "target_model_resnet56cifar10.pkl")))
        shadow_target_model.to(args.device)
        if not args.ensemble_method in ["pairwise_max"]:
            prepare_ensemble(preds_on_shadow_target_list_multiseed_avg, membership, args.ensemble_method, save_path, aux_set,
                             shadow_target_logits, args.device)

    elif args.mode == "run_ensemble":
        # read the predictions
        pred_path = args.preds_path
        base_preds = read_preds(pred_path, "", args.ensemble_seeds, args.dataset, args.target_model,
                                args.attacks, None)

        # read target data
        with open(os.path.join(args.target_data_path, "target_trainset.pkl"), "rb") as f:
            target_trainset = pickle.load(f)
        with open(os.path.join(args.target_data_path, "target_testset.pkl"), "rb") as f:
            target_testset = pickle.load(f)
        with open(os.path.join(args.target_data_path, "aux_set.pkl"), "rb") as f:
            aux_set = pickle.load(f)

        # run the ensemble (single seed)
        base_preds_list = base_preds[0]
        dataset_to_attack = ConcatDataset([target_trainset, target_testset])
        ensemble_result_path = os.path.join(args.ensemble_result_path, "single_seed")
        if not os.path.exists(ensemble_result_path):
            os.makedirs(ensemble_result_path)
        target_model = get_model(args.target_model, 10, 32)
        target_model.load_state_dict(
            torch.load(os.path.join(args.target_model_path, "target_model_resnet56cifar10.pkl")))
        target_model.to(args.device)
        target_model_logits = obtain_logits(target_model, dataset_to_attack, args.device)
        run_ensemble(base_preds_list, dataset_to_attack, args.ensemble_method, ensemble_result_path,
                     target_model_logits, args.ensemble_save_path + f"/{args.dataset}" + "/single_seed", args.device)


        # run the ensemble (multi seed)
        base_preds_list = multi_seed_avg(base_preds)
        ensemble_result_path_multi = os.path.join(args.ensemble_result_path, "multi_seed")
        if not os.path.exists(ensemble_result_path_multi):
            os.makedirs(ensemble_result_path_multi)
        target_model = get_model(args.target_model, 10, 32)
        target_model.load_state_dict(
            torch.load(os.path.join(args.target_model_path, "target_model_resnet56cifar10.pkl")))
        target_model.to(args.device)
        target_model_logits = obtain_logits(target_model, dataset_to_attack, args.device)
        if not args.ensemble_method in ["pairwise_max"]:
            run_ensemble(base_preds_list, dataset_to_attack, args.ensemble_method, ensemble_result_path_multi,
                         target_model_logits, args.ensemble_save_path + f"/{args.dataset}" + "/multi_seed", args.device)


    elif args.mode == "evaluation":
        # read target data
        with open(os.path.join(args.target_data_path, "target_trainset.pkl"), "rb") as f:
            target_trainset = pickle.load(f)
        with open(os.path.join(args.target_data_path, "target_testset.pkl"), "rb") as f:
            target_testset = pickle.load(f)
        with open(os.path.join(args.target_data_path, "aux_set.pkl"), "rb") as f:
            aux_set = pickle.load(f)
        index_to_data, membership = load_target_dataset(args.target_data_path)

        # read original preds
        pred_path = args.preds_path
        base_preds = read_preds(pred_path, "", args.ensemble_seeds, args.dataset, args.target_model,
                                args.attacks, membership)
        base_preds_list = base_preds[0]

        # read ensemble preds (single seed)
        ensemble_result_path = args.ensemble_result_path + "/single_seed"
        methods_names_single_seed, file_names = get_ensemble_methods(ensemble_result_path)
        methods_names_single_seed = ["single_seed_" + x for x in methods_names_single_seed]
        ensemble_preds_single_seed = []
        for fn in file_names:
            with open(os.path.join(ensemble_result_path, fn), "rb") as f:
                ensemble_preds_single_seed.append(prediction.Predictions(pickle.load(f), membership, fn))

        # read ensemble preds (multi seed)
        ensemble_result_path = args.ensemble_result_path + "/multi_seed"
        methods_names_multi_seed, file_names = get_ensemble_methods(ensemble_result_path)
        methods_names_multi_seed = ["multi_seed_" + x for x in methods_names_multi_seed]
        ensemble_preds_multi_seed = []
        for fn in file_names:
            with open(os.path.join(ensemble_result_path, fn), "rb") as f:
                ensemble_preds_multi_seed.append(prediction.Predictions(pickle.load(f), membership, fn))

        # combine single attack result with ensemble result
        name_list = args.attacks + methods_names_single_seed + methods_names_multi_seed
        preds_list = base_preds_list + ensemble_preds_single_seed + ensemble_preds_multi_seed

        # auc
        prediction.plot_auc(preds_list, name_list, f"{args.dataset} {args.target_model} ensemble AUC",
                            save_path=args.ensemble_result_path + '/ensemble_roc.png')

        # printing some stats
        balanced_acc = []
        acc = []

        for attack_name, pred in zip(name_list, preds_list):
            acc.append(pred.accuracy())
            balanced_acc.append(pred.balanced_attack_accuracy())

        for i in range(len(name_list)):
            print(f"{name_list[i]}: \tacc: {acc[i]:.4f}, \tbalanced acc: {balanced_acc[i]:.4f}")
