from miae.attacks.shokri_mia import *
import numpy as np


class TopKShokriAuxiliaryInfo(ShokriAuxiliaryInfo):
    def __init__(self, config, attack_model=AttackMLP):
        super().__init__(config, attack_model)
        # Initialize the top-k parameter
        self.top_k = config.get("top_k", 3)
        # rename their logger name
        self.log_path = config.get('log_path', None)
        if self.log_path is not None:
            self.logger = logging.getLogger('top_k_shokri_logger')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/top_k_shokri.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


TopKShokriModelAccess = ShokriModelAccess


class TopKShokriUtil(ShokriUtil):
    @classmethod
    def zero_mask_except_top_x(cls, data: np.ndarray, top: int) -> np.ndarray:
        """
        This method masks all elements to 0 in the input 2d array except the top 'n' elements in each entry.
        """
        mask = np.zeros_like(data)
        no_mask_index = np.argpartition(data, -top, axis=1)[:, -top:]
        rows = np.arange(data.shape[0])[:, None]
        mask[rows, no_mask_index] = data[rows, no_mask_index]
        return mask


class TopKShokriAttack(ShokriAttack):
    def __init__(self, target_model_access: TopKShokriModelAccess, auxiliary_info: TopKShokriAuxiliaryInfo,
                 target_data=None):
        super().__init__(target_model_access, auxiliary_info, target_data)
        self.topk = auxiliary_info.top_k

    def prepare(self, aux_dataset):
        """
        Prepare the attack. Most part of the code is copied from the ShokriAttack class.
        :param aux_dataset: the auxiliary dataset
        """
        if self.prepared:
            print("The attack has already prepared!")
            return

        self.attack_model = self.aux_info.attack_model(self.aux_info.num_classes)
        self.attack_model_dict = {}

        ShokriUtil.log(self.aux_info, "Start preparing the attack...", print_flag=True)

        # set seed
        set_seed(self.aux_info.seed)

        # create shadow datasets
        sub_shadow_dataset_list = ShokriUtil.split_dataset(aux_dataset, self.aux_info.num_shadow_models)
        # log/print the shadow dataset sizes
        ShokriUtil.log(self.aux_info, f"Shadow dataset[0] size: {sub_shadow_dataset_list[0].__len__()}")

        # step 1: train shadow models
        if not os.path.exists(self.aux_info.attack_dataset_path):
            # if attack dataset exists, then there's no need to retrain shadow models
            in_prediction_set_pred = None
            in_prediction_set_label = None
            out_prediction_set_pred = None
            out_prediction_set_label = None

            for i in range(self.aux_info.num_shadow_models):
                # train k shadows models to build attack dataset
                model_name = f"shadow_model_{i}.pt"
                model_path = os.path.join(self.aux_info.shadow_model_path, model_name)

                shadow_model_i = self.target_model_access.get_untrained_model()
                shadow_model_i.to(self.aux_info.device)


                if self.aux_info.shadow_diff_init:
                    try:
                        set_seed((self.aux_info.seed + i)*100) # *100 to avoid overlapping of different instances
                        shadow_model_i.initialize_weights()
                        ShokriUtil.log(self.aux_info, f"Shadow model initialized with seed: {(self.aux_info.seed + i)*100}")
                    except:
                        raise NotImplementedError("the model doesn't have .initialize_weights method")

                train_len = int(len(sub_shadow_dataset_list[i]) * self.aux_info.shadow_train_ratio)
                test_len = len(sub_shadow_dataset_list[i]) - train_len
                shadow_train_dataset, shadow_test_dataset = torch.utils.data.random_split(sub_shadow_dataset_list[i],
                                                                                          [train_len, test_len])

                shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=self.aux_info.shadow_batch_size,
                                                 shuffle=True)
                shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=self.aux_info.shadow_batch_size,
                                                shuffle=False)
                if os.path.exists(model_path):
                    ShokriUtil.log(self.aux_info,
                                   f"Loading shadow model {i + 1}/{self.aux_info.num_shadow_models}...")
                    shadow_model_i.load_state_dict(torch.load(model_path))
                else:
                    ShokriUtil.log(self.aux_info,
                                   f"Training shadow model {i + 1}/{self.aux_info.num_shadow_models}...")
                    shadow_model_i = ShokriUtil.train_shadow_model(shadow_model_i, shadow_train_loader,
                                                                   shadow_test_loader,
                                                                   self.aux_info)
                    torch.save(shadow_model_i.state_dict(), model_path)

                # building the attack dataset
                for data, target in shadow_train_loader:
                    data, target = data.to(self.aux_info.device), target.to(self.aux_info.device)
                    output = shadow_model_i(data)
                    if in_prediction_set_pred is None:  # first entry
                        in_prediction_set_pred = output.cpu().detach().numpy()
                        in_prediction_set_label = target.cpu().detach().numpy()
                    else:
                        in_prediction_set_pred = np.concatenate((in_prediction_set_pred, output.cpu().detach().numpy()))
                        in_prediction_set_label = np.concatenate(
                            (in_prediction_set_label, target.cpu().detach().numpy()))

                for data, target in shadow_test_loader:
                    data, target = data.to(self.aux_info.device), target.to(self.aux_info.device)
                    output = shadow_model_i(data)
                    if out_prediction_set_pred is None:  # first entry
                        out_prediction_set_pred = output.cpu().detach().numpy()
                        out_prediction_set_label = target.cpu().detach().numpy()
                    else:
                        out_prediction_set_pred = np.concatenate(
                            (out_prediction_set_pred, output.cpu().detach().numpy()))
                        out_prediction_set_label = np.concatenate(
                            (out_prediction_set_label, target.cpu().detach().numpy()))

            # step 2: create attack dataset for attack model training
            # This part is different from the original ShokriAttack class
            in_prediction_set_membership = np.ones(len(in_prediction_set_pred))
            out_prediction_set_membership = np.zeros(len(out_prediction_set_pred))

            # combine in and out prediction sets
            prediction_set_pred = np.concatenate((in_prediction_set_pred, out_prediction_set_pred))
            prediction_set_pred = TopKShokriUtil.zero_mask_except_top_x(prediction_set_pred, self.topk)
            prediction_set_label = np.concatenate((in_prediction_set_label, out_prediction_set_label))
            prediction_set_membership = np.concatenate((in_prediction_set_membership, out_prediction_set_membership))

            # shuffle the prediction set
            shuffle_idx = np.arange(len(prediction_set_pred))
            np.random.shuffle(shuffle_idx)
            prediction_set_pred = prediction_set_pred[shuffle_idx]
            prediction_set_label = prediction_set_label[shuffle_idx]
            prediction_set_membership = prediction_set_membership[shuffle_idx]

            # build the dataset for attack model training
            self.attack_dataset = AttackTrainingSet(prediction_set_pred, prediction_set_label,
                                                    prediction_set_membership)
            torch.save(self.attack_dataset, self.aux_info.attack_dataset_path)

        # step 3: train attack model
        set_seed(self.aux_info.seed)
        self.attack_dataset = torch.load(self.aux_info.attack_dataset_path)
        train_len = int(len(self.attack_dataset) * self.aux_info.attack_train_ratio)
        test_len = len(self.attack_dataset) - train_len

        if self.aux_info.attack_train_ratio < 1.0:
            attack_train_dataset, attack_test_dataset = torch.utils.data.random_split(self.attack_dataset,
                                                                                      [train_len, test_len])
        else:
            attack_train_dataset = self.attack_dataset
            attack_test_dataset = None

        # train attack models for each label
        labels = np.unique(self.attack_dataset.class_labels)
        # if attack model exists, then there's no need to retrain attack models
        if len(labels) == len(os.listdir(self.aux_info.attack_model_path)):
            ShokriUtil.log(self.aux_info, "Loading attack models...")
            for i, label in enumerate(labels):
                model = self.aux_info.attack_model(self.aux_info.num_classes)
                model.load_state_dict(torch.load(f"{self.aux_info.attack_model_path}/attack_model_{label}.pt"))
                model.to(self.aux_info.device)
                self.attack_model_dict[label] = model
        else:
            for i, label in enumerate(labels):
                ShokriUtil.log(self.aux_info, f"Training attack model for {i}/{len(labels)} label \"{label}\" ..."
                               , print_flag=True)
                # filter the dataset with the label
                attack_train_dataset_filtered = ShokriUtil.filter_dataset(attack_train_dataset, label)
                attack_test_dataset_filtered = ShokriUtil.filter_dataset(attack_test_dataset,
                                                                         label) if attack_test_dataset else None
                self.attack_train_loader = DataLoader(attack_train_dataset_filtered,
                                                      batch_size=self.aux_info.attack_batch_size,
                                                      shuffle=True)
                self.attack_test_loader = DataLoader(attack_test_dataset_filtered,
                                                     batch_size=self.aux_info.attack_batch_size,
                                                     shuffle=True) if attack_test_dataset else None
                untrained_attack_model = self.aux_info.attack_model(self.aux_info.num_classes)
                untrained_attack_model.to(self.aux_info.device)
                trained_attack_model = ShokriUtil.train_attack_model(untrained_attack_model, self.attack_train_loader,
                                                                     self.attack_test_loader,
                                                                     self.aux_info)
                self.attack_model_dict[label] = trained_attack_model
                torch.save(trained_attack_model.state_dict(),
                           f"{self.aux_info.attack_model_path}/attack_model_{label}.pt")

        self.prepared = True

    def infer(self, target_data) -> np.ndarray:
        """
        Infer the membership of the target data. Most part of the code is copied from the ShokriAttack class.
        """
        if not self.prepared:
            raise ValueError("The attack has not been prepared!")

        # load the attack models
        labels = np.unique(self.attack_dataset.class_labels)
        for label in labels:
            if label not in self.attack_model_dict:
                model = self.aux_info.attack_model(self.topk)
                model.load_state_dict(
                    torch.load(f"{self.aux_info.attack_model_path}/attack_model_{label}.pt"))
                model.to(self.aux_info.device)
                self.attack_model_dict[label] = model

        # infer the membership
        self.target_model_access.model.to(self.aux_info.device)
        membership = []

        target_data_loader = DataLoader(target_data, batch_size=self.aux_info.batch_size, shuffle=False)
        ShokriUtil.log(self.aux_info, "Inferencing the membership of the target data...")
        for data, target in tqdm(target_data_loader):
            data = data.to(self.aux_info.device)
            output = self.target_model_access.model(data)
            output = output.cpu().detach().numpy()
            output = TopKShokriUtil.zero_mask_except_top_x(np.array(output), self.topk)
            for i, label in enumerate(target):
                label = label.item()
                member_pred = self.attack_model_dict[label](torch.tensor(output[i]).to(self.aux_info.device))
                member_pred = member_pred.cpu().detach().numpy()
                membership.append(member_pred.reshape(-1))

        return np.array(np.transpose(membership)[1])

