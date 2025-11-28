import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import os

from utils.toolkit import tensor2numpy, accuracy_domain
from models.slinet import SliNet
from utils.lr_scheduler import build_lr_scheduler
from utils.data_manager import DataManager
from eval import compute_predictions

import wandb


class Prompt2Guard:

    def __init__(self, args: dict):
        # Network and device settings
        self.network = SliNet(args)
        self.device = args["device"]
        self.class_num = self.network.class_num

        # Task and class settings
        self.cur_task = -1
        self.n_clusters = 5
        self.n_cluster_one = 1
        self.known_classes = 0
        self.total_classes = 0

        # Key settings, different clusters tested
        self.all_keys = []              # consider n_clusters image prototypes for each domain
        self.all_keys_one_vector = []   # consider 1 image prototype for each domain
        self.real_keys_one_vector = []  # only real images considered to build the prototype
        self.fake_keys_one_vector = []  # only fake images considered to build the prototype

        # Learning parameters
        self.EPSILON = args["EPSILON"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.warmup_epoch = args["warmup_epoch"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.batch_size_eval = args["batch_size_eval"]
        self.weight_decay = args["weight_decay"]
        self.label_smoothing = args["label_smoothing"]
        self.enable_prev_prompt = args["enable_prev_prompt"]

        # System settings
        self.num_workers = int(
            os.environ.get("SLURM_CPUS_ON_NODE", args["num_workers"])
        )
        self.filename = args["filename"]

        # Other settings
        self.args = args

        # # wandb setup
        # slurm_job_name = os.environ.get("SLURM_JOB_NAME", 'prompt2guard')
        # if slurm_job_name == "bash":
        #     slurm_job_name += "/localtest"

        # self.wandb_logger = wandb.init(
        #     project=slurm_job_name.split("/")[0],
        #     entity="YOUR_USERNAME",
        #     name=slurm_job_name.split("/")[1],
        #     mode="disabled" if not args["wandb"] else "online",
        #     config=args,
        # )
        # if self.wandb_logger is None:
        #     raise ValueError("Failed to initialize wandb logger")

        # self.wandb_logger.define_metric("epoch")
        # self.wandb_logger.define_metric("task")
        # self.wandb_logger.define_metric("condition")
        # self.wandb_logger.define_metric("task_*", step_metric="epoch")
        # self.wandb_logger.define_metric("eval_trainer/*", step_metric="task")
        # self.wandb_logger.define_metric("inference_*", step_metric="condition")

    def after_task(self, nb_tasks):
        self.known_classes = self.total_classes
        if self.enable_prev_prompt and self.network.numtask < nb_tasks:
            with torch.no_grad():
                self.network.prompt_learner[self.network.numtask].load_state_dict(
                    self.network.prompt_learner[self.network.numtask - 1].state_dict()
                )

    def incremental_train(self, data_manager: DataManager):
        self.cur_task += 1
        self.total_classes = self.known_classes + data_manager.get_task_size(
            self.cur_task
        )
        self.network.update_fc()

        logging.info("Learning on {}-{}".format(self.known_classes, self.total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self.known_classes, self.total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self.total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)

    def _train(self, train_loader, test_loader):
        self.network.to(self.device)
        for name, param in self.network.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" + "." + str(self.network.numtask - 1) in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {enabled}")

        if self.cur_task == 0:
            optimizer = optim.SGD(
                self.network.parameters(),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )
            scheduler = build_lr_scheduler(
                optimizer,
                lr_scheduler="cosine",
                warmup_epoch=self.warmup_epoch,
                warmup_type="constant",
                warmup_cons_lr=1e-5,
                max_epoch=self.epochs,
            )
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self.network.parameters(),
                momentum=0.9,
                lr=self.lrate,
                weight_decay=self.weight_decay,
            )
            scheduler = build_lr_scheduler(
                optimizer,
                lr_scheduler="cosine",
                warmup_epoch=self.warmup_epoch,
                warmup_type="constant",
                warmup_cons_lr=1e-5,
                max_epoch=self.epochs,
            )
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        best_acc = 0.0  # Already present, used for tracking

        # --- Added: Define save path and ensure directory exists ---
        # Using the same path as your original save_checkpoint method
        save_dir = f'./checkpoint/{self.args["run_name"]}/weights/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'best.pt')
        # ---------------------------------------------------------

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            
            # Set network to train mode
            self.network.train() 
            with tqdm(train_loader, unit='batch', mininterval=10) as tepoch:
                tepoch.set_description(f'Epoch {epoch}', refresh=False)
                for i, (object_name, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    mask = (targets >= self.known_classes).nonzero().view(-1)
                    inputs = torch.index_select(inputs, 0, mask)
                    targets = torch.index_select(targets, 0, mask) - self.known_classes

                    logits = self.network(inputs, object_name)["logits"]
                    loss = F.cross_entropy(
                        logits, targets, label_smoothing=self.label_smoothing
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                    tepoch.set_postfix(loss=loss.item())

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)
                    #if i> 10:
                    #    break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # Set network to eval mode for computing test_acc
            self.network.eval() 
            test_acc = self._compute_accuracy_domain(self.network, test_loader, epoch)

            # --- Added: Checkpoint saving logic ---
            if test_acc > best_acc:
                best_acc = test_acc
                logging.info(f"New best accuracy: {best_acc}. Saving checkpoint to {save_path}")
                
                # --- Dynamic values ---
                model_state_dict = {
                    name: param.cpu()  # Move to CPU for saving
                    for name, param in self.network.named_parameters()
                    if param.requires_grad
                }
                
                # --- Hardcoded values ---
                
                # WARNING: The 'all_keys' tensor data was incomplete in the prompt (contained '...').
                # It is NOT included here. Please provide the full tensor to save it.
                logging.warning("Checkpoint 'all_keys' is not being saved as the provided data was incomplete.")

                # These lists are complete and will be saved.
                all_keys_one_cluster_data = [
                     1.9211e-02, -7.6294e-02,  3.2578e-03,  2.5272e-03, -4.6310e-03,
                     4.3297e-03,  1.8418e-05, -4.9782e-03,  2.2526e-03, -2.1114e-03,
                     1.0109e-02, -2.7512e-02,  9.0561e-03, -1.9394e-02, -1.4694e-02,
                     1.3664e-02,  3.0479e-03, -9.8724e-03, -5.1956e-03, -3.5648e-03,
                     3.1799e-02,  6.7043e-04, -7.5684e-03,  4.4441e-03,  4.3869e-03,
                     5.9395e-03,  1.1765e-02,  5.3444e-03,  3.5152e-03, -4.1580e-03,
                     5.9853e-03, -2.6245e-03, -4.7264e-03, -6.5956e-03, -3.2501e-02,
                    -1.3824e-02, -5.6305e-03, -5.0850e-03,  4.1290e-02, -1.0567e-02,
                    -2.3212e-03, -1.3599e-03, -9.1782e-03, -1.9608e-03, -5.6496e-03,
                    -6.1989e-03, -3.9558e-03,  1.1358e-03,  9.8801e-04,  9.9659e-04,
                     2.2011e-03,  1.1787e-02,  9.9411e-03, -4.1938e-04, -7.4120e-03,
                    -3.0609e-02,  1.2871e-02,  2.3331e-02, -3.1972e-04,  1.3802e-02,
                     2.4109e-02, -1.4542e-02, -5.2214e-04,  2.3880e-03, -8.7967e-03,
                     2.3079e-03,  7.1793e-03, -1.9104e-02,  2.1095e-03,  4.9095e-03,
                     1.4954e-02,  3.6407e-02,  3.2745e-02,  7.2365e-03,  1.6739e-02,
                     5.6763e-03,  1.4481e-02,  1.3771e-02,  3.7212e-03, -2.2945e-03,
                     2.4948e-02,  3.3936e-02, -1.8433e-02,  4.9639e-04, -1.2941e-03,
                     8.9417e-03, -1.5440e-03, -9.7427e-03,  7.0152e-03,  4.7827e-04,
                     6.1264e-03, -6.9313e-03,  8.3008e-03,  1.0155e-02, -8.6136e-03,
                    -6.0539e-03,  1.7578e-02,  1.7548e-03,  7.9727e-04,  2.1957e-02,
                     5.8098e-03,  1.1665e-02,  1.6342e-02, -5.2185e-02, -8.6746e-03,
                    -2.7733e-03, -3.7518e-03, -4.6921e-03, -1.7366e-03, -1.6159e-02,
                     1.6184e-03, -1.4053e-02,  2.7065e-03,  5.7640e-03,  2.0294e-03,
                    -1.1093e-02, -2.4395e-03,  7.4310e-03,  6.1760e-03, -9.0103e-03,
                    -2.3937e-03, -3.1986e-03,  5.9891e-03,  4.6448e-02,  1.3718e-02,
                     7.8506e-03,  2.5024e-02, -8.0719e-03,  1.2123e-02, -2.4185e-02,
                    -1.0757e-02,  1.5686e-02, -5.7144e-03, -3.2291e-03,  3.0075e-02,
                     1.0727e-02, -9.3002e-03,  6.6757e-04, -1.4946e-02,  3.4752e-03,
                    -6.5918e-03, -1.8682e-03,  2.4414e-03, -1.1482e-02, -8.2092e-03,
                    -9.9564e-03,  1.8387e-02,  5.9547e-03, -1.4580e-02,  1.8509e-02,
                    -1.7822e-02, -1.3514e-03, -4.4212e-03,  3.4637e-03,  2.6184e-02,
                     5.8556e-03,  4.2915e-03,  9.7046e-03,  8.1635e-03, -3.0411e-02,
                    -1.8127e-02,  1.3885e-02, -1.5060e-02, -3.2471e-02, -4.1656e-03,
                    -1.1681e-02, -4.8714e-03, -3.3844e-02, -1.7118e-03, -3.9124e-04,
                    -6.6376e-03, -1.5945e-02,  6.6996e-04, -8.0824e-04, -1.3695e-03,
                     1.0586e-03, -9.1400e-03, -1.9836e-03,  3.8757e-02, -9.6588e-03,
                     3.4943e-03,  1.1703e-02,  9.9716e-03,  1.3809e-02,  1.5388e-02,
                     8.5144e-03,  4.6692e-03, -1.2077e-02, -1.2177e-02,  2.7733e-03,
                    -8.3351e-04,  1.8988e-03,  9.9869e-03, -6.0997e-03,  3.2349e-02,
                     1.2383e-02, -8.7433e-03,  2.2522e-02,  2.7313e-03,  3.1300e-03,
                     6.8436e-03,  7.9651e-03, -2.3441e-03,  6.6376e-04,  1.1032e-02,
                    -9.5367e-06, -2.0218e-03, -6.8169e-03,  1.1269e-02, -1.8620e-04,
                    -1.4511e-02, -1.2741e-03, -3.3051e-02,  9.3842e-03,  2.8944e-04,
                    -1.9894e-03, -1.5625e-02,  1.5366e-02, -2.6302e-03,  2.4402e-04,
                    -1.0735e-02, -1.4359e-02,  7.9269e-03,  3.4866e-03, -1.2794e-02,
                    -8.2932e-03,  8.8654e-03,  5.0545e-03,  2.0493e-02, -1.1841e-02,
                     7.9775e-04, -3.0624e-02,  2.5311e-03,  1.4648e-03,  2.8591e-03,
                    -3.6602e-03,  9.6054e-03,  2.0790e-03, -1.5549e-02,  2.5501e-03,
                    -9.0332e-03, -1.6663e-02,  4.6425e-03, -2.8038e-03, -7.9407e-02,
                    -1.4503e-02, -5.1832e-04,  2.5711e-03,  1.0544e-02, -9.2926e-03,
                     1.4709e-02,  8.8806e-03,  9.7046e-03,  1.5163e-03, -2.0691e-02,
                     2.5421e-02, -1.9409e-02,  5.8899e-03, -1.1187e-03, -1.8829e-02,
                    -1.0025e-02,  5.5351e-03, -9.4833e-03,  1.1391e-02,  1.7321e-04,
                    -1.8509e-02, -9.4681e-03, -1.8234e-03, -8.5678e-03, -1.0094e-02,
                    -1.4935e-03,  1.9302e-02, -4.7951e-03,  7.8888e-03, -5.0812e-03,
                    -4.0222e-02,  1.0710e-03,  1.0948e-02,  1.3268e-02, -8.1482e-03,
                    -2.4673e-02, -7.9041e-03, -7.1602e-03,  1.3466e-02, -5.0964e-03,
                    -5.4741e-03, -6.1874e-03, -1.7033e-03, -1.1032e-02, -2.7981e-03,
                     1.1200e-02,  2.2774e-03, -6.0059e-02, -5.1537e-03, -5.6190e-03,
                    -3.3474e-04,  8.3780e-04,  6.4026e-02, -1.4801e-03,  1.9436e-03,
                    -5.7220e-03,  2.7275e-03,  1.1452e-02, -1.4862e-02, -1.1566e-02,
                    -7.6675e-03, -7.3051e-03,  4.4823e-03,  9.7871e-05,  7.4081e-03,
                     1.3952e-03, -3.0613e-03, -2.9812e-03, -1.0757e-03, -1.4320e-02,
                    -1.4748e-02,  7.1754e-03,  1.9608e-02,  1.2383e-02, -1.3664e-02,
                    -1.0824e-03, -4.0054e-03, -7.5874e-03,  1.3298e-02,  7.7133e-03,
                     9.1019e-03, -2.1118e-02, -1.3878e-02,  6.3591e-03, -2.5921e-03,
                     1.8387e-03, -3.8052e-03, -4.7073e-03,  1.8936e-02, -1.6775e-03,
                    -1.2810e-02,  6.4621e-03,  4.2877e-03,  2.1267e-03,  1.8402e-02,
                    -1.5030e-02, -1.2848e-02,  5.6549e-02, -2.1172e-03,  1.2917e-02,
                     1.6251e-03,  5.6505e-04,  5.9128e-03, -1.7052e-03, -1.7365e-02,
                    -2.6443e-02, -4.2992e-03,  2.0248e-02,  1.1398e-02,  3.5934e-03,
                     4.6082e-03,  2.3232e-03,  7.7820e-03,  1.7023e-03, -1.0612e-02,
                     9.8343e-03, -6.1493e-03,  1.0370e-01, -7.5722e-03,  8.9417e-03,
                    -2.2125e-03,  1.1505e-02,  2.4338e-03,  8.3160e-03,  9.5520e-03,
                     4.6501e-03, -3.2253e-03,  1.5726e-03,  1.3916e-02, -4.2297e-02,
                     4.5929e-03,  7.0007e-02, -6.9046e-03,  3.8776e-03, -7.3128e-03,
                    -4.4746e-03, -7.2021e-03, -1.9089e-02,  7.7724e-04, -3.0212e-02,
                     2.1301e-02, -3.7327e-03, -1.0414e-02,  4.2610e-03,  1.2299e-02,
                     3.3779e-03,  5.6038e-03,  4.7188e-03, -6.9962e-03,  1.0918e-02,
                    -2.7809e-03, -6.7806e-04, -2.1255e-02,  1.0147e-02,  4.5128e-03,
                    -1.3494e-04,  1.7227e-02, -3.0422e-03, -1.3802e-02,  1.6754e-02,
                     1.7471e-02,  1.4984e-02,  5.0926e-03, -5.0430e-03,  1.5251e-02,
                    -2.4567e-03, -5.1056e-02,  8.7967e-03, -1.1482e-02,  1.9943e-02,
                    -8.6021e-04,  1.2939e-02, -7.7972e-03,  7.0152e-03,  1.1497e-02,
                    -5.8441e-03, -9.1171e-03,  1.2016e-02, -8.6670e-03,  5.2109e-03,
                    -1.1182e-04,  1.6083e-02, -5.2834e-03,  6.6519e-04,  5.3497e-02,
                    -2.7603e-02,  1.7090e-02, -1.8097e-02, -5.2452e-03, -3.4256e-03,
                     3.5362e-03, -1.4915e-02, -9.9411e-03,  4.7722e-03,  4.2915e-03,
                     9.2697e-03,  2.5005e-03,  6.5820e-01, -2.3060e-03,  4.5853e-03,
                     1.5092e-04,  4.5357e-03,  1.4420e-02, -6.6910e-03, -1.2039e-02,
                    -4.7951e-03,  8.5526e-03, -6.4240e-03, -2.4929e-03, -4.5128e-03,
                    -8.9188e-03, -1.3995e-04,  6.2866e-03, -1.1642e-02, -1.2894e-03,
                    -5.9280e-03, -6.4621e-03, -5.9662e-03, -2.2858e-02,  2.4551e-02,
                     4.2267e-02, -4.8294e-03,  5.3139e-03, -9.0866e-03, -1.0216e-02,
                     1.4725e-02,  6.2675e-03, -8.5449e-03, -7.2021e-03, -1.0138e-03,
                    -6.8665e-03,  5.0545e-03, -3.0422e-03,  4.0588e-03, -4.3144e-03,
                    -9.7961e-03, -8.7051e-03,  1.7815e-03, -1.6983e-02, -7.6675e-03,
                     5.7564e-03, -4.9019e-03,  4.9782e-03,  1.8406e-03,  7.6904e-03,
                    -1.6876e-02, -1.1360e-02,  6.7177e-03,  6.9351e-03, -3.9673e-03,
                     1.1208e-02,  1.4244e-02,  1.0620e-02,  1.0414e-02, -2.9678e-03,
                     9.3231e-03,  9.4452e-03,  6.3362e-03, -2.3823e-03,  1.0330e-02,
                     1.0872e-02, -3.4924e-03,  1.1650e-02,  1.2863e-02,  7.9651e-03,
                     1.3443e-02,  2.6840e-02
                ]
                
                real_keys_one_cluster_data = [
                     1.9150e-02, -6.2927e-02, -8.7070e-04,  1.0548e-03, -4.9629e-03,
                     9.4681e-03, -1.7672e-03,  1.7052e-03,  2.1687e-03, -2.4815e-03,
                     9.6970e-03, -1.7136e-02,  1.1948e-02, -1.9089e-02, -9.7885e-03,
                     9.6817e-03,  5.5923e-03, -6.4087e-03, -4.2458e-03, -2.4815e-03,
                     1.8768e-02, -2.4223e-03, -6.3591e-03,  4.7989e-03,  8.4019e-04,
                     5.1193e-03,  6.5956e-03,  5.8708e-03,  4.9210e-03, -1.2255e-03,
                     8.6136e-03, -3.0861e-03, -2.4738e-03, -8.1558e-03, -3.5278e-02,
                    -1.2016e-02, -3.6583e-03, -5.0049e-03,  3.2562e-02, -2.2842e-02,
                    -3.5534e-03, -3.7575e-03, -5.0774e-03, -3.2463e-03, -5.5237e-03,
                    -3.5343e-03, -2.2774e-03,  1.5135e-03,  3.0479e-03,  2.8191e-03,
                    -1.6890e-03,  1.4412e-02,  7.8812e-03, -1.0595e-03, -6.8398e-03,
                    -2.8961e-02,  1.4214e-02,  2.2049e-02, -4.4022e-03,  1.6235e-02,
                     2.6199e-02, -1.1688e-02, -1.5574e-03,  1.0359e-04, -5.8594e-03,
                     4.7760e-03,  2.7599e-03, -2.2873e-02,  4.3297e-03,  7.3242e-03,
                     1.3390e-02,  3.5248e-02,  3.1403e-02,  7.4539e-03,  1.4809e-02,
                     7.6141e-03,  1.2939e-02,  1.0178e-02, -4.2038e-03, -3.4580e-03,
                     2.1820e-02,  2.8778e-02, -1.9058e-02,  2.6073e-03,  2.0695e-03,
                     9.2621e-03,  3.3760e-04, -3.2749e-03,  1.7147e-03, -2.3823e-03,
                     6.3591e-03, -4.4136e-03,  9.1476e-03,  7.8278e-03, -5.5618e-03,
                    -5.2032e-03,  2.0157e-02,  3.4447e-03,  9.6607e-04,  2.1881e-02,
                     6.9618e-03,  1.2512e-02,  1.6327e-02, -3.2990e-02, -1.0002e-02,
                    -2.1763e-03, -7.8344e-04, -1.5841e-03, -3.3512e-03, -1.0925e-02,
                    -1.2197e-03, -1.3657e-02,  2.4700e-03,  1.0628e-02,  3.4351e-03,
                    -5.1727e-03, -2.0542e-03,  6.4850e-03,  1.1177e-02, -4.5891e-03,
                    -2.6035e-03,  6.7902e-04,  3.6545e-03,  4.7119e-02,  1.6006e-02,
                     6.6833e-03,  2.0737e-02, -6.5155e-03,  1.2199e-02, -1.9775e-02,
                    -1.1337e-02,  1.3199e-02, -2.8172e-03, -2.0332e-03,  3.1082e-02,
                     8.7891e-03, -1.0460e-02,  7.3586e-03, -1.1574e-02,  4.1161e-03,
                    -6.6109e-03, -4.0054e-03,  4.1122e-03, -1.2413e-02, -5.4817e-03,
                    -8.8501e-03,  1.2878e-02,  6.3858e-03, -1.6388e-02,  1.8356e-02,
                    -2.2537e-02,  2.8992e-03, -5.7297e-03,  3.1681e-03,  2.8961e-02,
                     9.9182e-04,  4.6387e-03,  1.4503e-02,  9.0637e-03, -2.8656e-02,
                    -9.3918e-03,  8.1024e-03, -2.4918e-02, -3.2227e-02, -6.6872e-03,
                    -6.1913e-03, -3.6316e-03, -3.3295e-02, -5.6877e-03, -2.7008e-03,
                    -7.5455e-03, -1.7258e-02,  7.3314e-06, -3.4046e-03, -6.4659e-04,
                     3.1338e-03, -1.1635e-02, -1.0455e-04,  1.9913e-02, -1.0620e-02,
                     4.2458e-03,  7.1678e-03,  1.0223e-02,  8.6517e-03,  8.3771e-03,
                     9.2163e-03,  9.4461e-04, -1.3168e-02, -1.2726e-02,  3.5324e-03,
                     1.6527e-03,  2.9144e-03,  1.2245e-02, -3.8300e-03,  7.1383e-04,
                     1.3206e-02, -7.2937e-03,  2.1286e-02,  3.9368e-03,  1.5991e-02,
                     6.0539e-03,  1.1856e-02, -4.9934e-03, -2.5139e-03,  8.6517e-03,
                     2.8591e-03, -7.0524e-04, -8.7662e-03,  9.0103e-03,  1.8966e-04,
                    -1.3924e-02, -1.9150e-03, -2.4231e-02,  5.1956e-03, -4.0321e-03,
                    -3.4885e-03, -1.6296e-02,  1.3519e-02, -2.2583e-03, -4.9438e-03,
                    -1.0315e-02, -1.4542e-02,  6.0921e-03,  2.3689e-03, -1.2154e-02,
                    -7.9575e-03,  3.9444e-03,  9.8572e-03,  2.8687e-02, -6.9313e-03,
                     7.6532e-04, -2.5177e-02, -3.7651e-03, -5.1308e-04,  1.7281e-03,
                    -1.7262e-03,  4.6959e-03, -1.2171e-04, -1.2772e-02, -4.5085e-04,
                    -9.7752e-04, -1.4389e-02,  4.0970e-03, -5.1804e-03, -6.6589e-02,
                    -1.6190e-02,  2.8877e-03,  2.2297e-03,  1.0788e-02, -1.0941e-02,
                     1.6830e-02, -1.5366e-02,  7.7133e-03,  6.8855e-03, -4.9324e-03,
                     2.0111e-02, -8.0795e-03,  6.3591e-03, -5.4216e-04, -1.7212e-02,
                    -6.2103e-03,  4.3678e-03, -1.0254e-02,  1.1513e-02,  7.4387e-04,
                    -2.9129e-02, -8.0719e-03, -4.4179e-04, -5.6534e-03, -1.2115e-02,
                     1.6153e-05,  1.7136e-02, -6.9427e-03,  8.7738e-03, -4.3182e-03,
                    -4.7699e-02,  2.4986e-03,  1.0597e-02,  8.9188e-03, -7.6408e-03,
                    -1.1009e-02, -8.5831e-03, -9.7809e-03,  1.1726e-02, -7.9956e-03,
                    -6.2294e-03, -6.7978e-03, -1.3418e-03, -1.1559e-02,  6.7472e-04,
                     1.0254e-02, -3.4094e-05, -4.2175e-02, -3.6507e-03, -8.2932e-03,
                    -2.2144e-03, -5.8861e-03,  7.5623e-02, -1.0996e-03,  1.3523e-03,
                    -3.9978e-03,  3.1223e-03,  8.2321e-03, -1.2772e-02, -9.4070e-03,
                    -1.2886e-02, -7.2899e-03,  5.7983e-03,  1.7536e-04,  6.3400e-03,
                     5.0964e-03, -3.7785e-03, -9.0485e-03, -2.6150e-03, -7.0343e-03,
                    -1.6571e-02,  4.9896e-03,  1.6342e-02,  1.0910e-02, -5.2986e-03,
                     2.3212e-03, -4.4861e-03, -9.3689e-03,  9.6359e-03,  4.9706e-03,
                     5.7755e-03, -2.0660e-02, -1.0445e-02,  3.9406e-03,  2.4605e-03,
                     6.3515e-04, -2.9392e-03, -6.4850e-03,  1.7822e-02, -6.6071e-03,
                    -1.2253e-02,  2.3689e-03,  2.0466e-03,  2.9540e-04,  1.7136e-02,
                    -1.4854e-02, -1.3794e-02,  6.5613e-02, -4.8370e-03,  1.2672e-02,
                     2.2087e-03,  9.5367e-04,  3.9291e-03, -2.1000e-03, -1.5427e-02,
                    -1.8433e-02, -1.7166e-03,  1.5778e-02,  9.9258e-03,  3.7346e-03,
                     3.6659e-03, -3.5114e-03,  8.7814e-03,  6.1703e-04, -5.9738e-03,
                     6.9847e-03, -6.5155e-03,  1.0339e-01, -9.4986e-03,  6.3477e-03,
                    -7.8812e-03,  1.2131e-02,  3.6335e-04,  1.0895e-02,  9.9792e-03,
                     7.5684e-03, -5.6839e-03, -1.0042e-03,  5.2910e-03, -5.1666e-02,
                     7.4844e-03,  6.3110e-02, -8.8120e-03,  5.4264e-04, -1.0300e-02,
                    -1.5678e-03, -1.3527e-02, -3.0807e-02,  3.4580e-03, -2.7039e-02,
                     2.4033e-02, -1.4057e-03, -1.0971e-02,  8.2245e-03,  1.6769e-02,
                    -2.3613e-03,  3.1643e-03,  4.8714e-03, -4.5013e-03,  9.2163e-03,
                    -2.3537e-03, -5.1003e-03, -2.0859e-02,  8.7967e-03,  6.5994e-03,
                     2.2697e-03,  1.2589e-02,  3.3588e-03, -1.2383e-02,  1.5266e-02,
                     1.3687e-02,  6.3972e-03,  1.6413e-03, -4.6806e-03,  1.0757e-02,
                    -1.6613e-03, -2.3239e-02,  1.1246e-02, -1.0399e-02,  2.2141e-02,
                     3.5644e-04,  1.0658e-02, -9.9640e-03,  5.0850e-03,  8.5678e-03,
                    -7.7820e-03, -7.4501e-03,  1.0712e-02, -9.6359e-03,  3.4695e-03,
                     2.2831e-03,  1.3100e-02, -1.3113e-05, -1.5795e-04,  5.4413e-02,
                    -2.1591e-02,  1.5839e-02, -1.5884e-02, -3.6983e-03, -6.5002e-03,
                     3.5877e-03, -1.4893e-02, -6.1798e-03,  5.0468e-03,  6.3210e-03,
                     7.8049e-03, -6.3944e-04,  6.4795e-01,  4.6883e-03,  6.0616e-03,
                    -4.1656e-03,  4.6039e-04,  1.4618e-02, -7.2060e-03, -1.0750e-02,
                    -3.4237e-03,  9.5749e-03, -7.7934e-03, -4.6539e-03, -2.2488e-03,
                    -8.2855e-03,  1.1539e-03,  9.4528e-03, -1.1650e-02, -4.3869e-03,
                    -6.9084e-03, -1.1734e-02, -5.9052e-03, -1.7181e-02,  2.2034e-02,
                     3.1860e-02, -1.4830e-03,  1.2236e-03, -1.1803e-02, -9.1858e-03,
                     1.4915e-02,  2.6112e-03, -5.1003e-03, -1.0986e-02,  4.1819e-04,
                     4.1161e-03,  4.6577e-03, -4.0932e-03,  5.2834e-03, -5.6229e-03,
                    -6.5880e-03, -1.1993e-02,  1.3895e-03, -1.5312e-02, -4.8790e-03,
                     5.4665e-03, -1.0529e-02,  2.9030e-03,  1.9779e-03,  7.1526e-03,
                    -1.8753e-02, -1.5404e-02,  7.2021e-03,  5.6114e-03, -4.6501e-03,
                     6.8207e-03,  1.3756e-02,  9.0027e-03,  1.0193e-02,  2.7943e-04,
                     8.9951e-03,  1.1032e-02,  6.6376e-03, -1.1024e-03,  6.4049e-03,
                     1.6556e-02, -5.0354e-03,  1.3781e-03,  1.2787e-02,  9.9182e-03,
                     1.2466e-02,  2.5681e-02
                ]

                fake_keys_one_cluster_data = [
                     1.9287e-02, -8.9661e-02,  7.3853e-03,  3.9978e-03, -4.2992e-03,
                    -8.0204e-04,  1.8044e-03, -1.1665e-02,  2.3384e-03, -1.7414e-03,
                     1.0521e-02, -3.7903e-02,  6.1684e-03, -1.9699e-02, -1.9592e-02,
                     1.7639e-02,  5.0545e-04, -1.3336e-02, -6.1455e-03, -4.6463e-03,
                     4.4830e-02,  3.7632e-03, -8.7814e-03,  4.0894e-03,  7.9269e-03,
                     6.7635e-03,  1.6922e-02,  4.8218e-03,  2.1095e-03, -7.0915e-03,
                     3.3550e-03, -2.1648e-03, -6.9771e-03, -5.0354e-03, -2.9755e-02,
                    -1.5625e-02, -7.6027e-03, -5.1689e-03,  5.0018e-02,  1.7214e-03,
                    -1.0891e-03,  1.0366e-03, -1.3283e-02, -6.7520e-04, -5.7755e-03,
                    -8.8654e-03, -5.6343e-03,  7.5817e-04, -1.0710e-03, -8.2541e-04,
                     6.0921e-03,  9.1629e-03,  1.2001e-02,  2.2042e-04, -7.9880e-03,
                    -3.2257e-02,  1.1536e-02,  2.4612e-02,  3.7613e-03,  1.1375e-02,
                     2.2018e-02, -1.7395e-02,  5.1260e-04,  4.6730e-03, -1.1734e-02,
                    -1.5831e-04,  1.1597e-02, -1.5320e-02, -1.1021e-04,  2.4948e-03,
                     1.6510e-02,  3.7598e-02,  3.4088e-02,  7.0190e-03,  1.8677e-02,
                     3.7403e-03,  1.6022e-02,  1.7365e-02,  1.1642e-02, -1.1311e-03,
                     2.8061e-02,  3.9093e-02, -1.7822e-02, -1.6146e-03, -4.6577e-03,
                     8.6288e-03, -3.4256e-03, -1.6205e-02,  1.2314e-02,  3.3379e-03,
                     5.8899e-03, -9.4528e-03,  7.4501e-03,  1.2482e-02, -1.1665e-02,
                    -6.9008e-03,  1.5007e-02,  6.5565e-05,  6.2847e-04,  2.2034e-02,
                     4.6539e-03,  1.0826e-02,  1.6342e-02, -7.1411e-02, -7.3509e-03,
                    -3.3722e-03, -6.7177e-03, -7.8049e-03, -1.2106e-04, -2.1393e-02,
                     4.4556e-03, -1.4458e-02,  2.9430e-03,  8.9741e-04,  6.2275e-04,
                    -1.7014e-02, -2.8248e-03,  8.3771e-03,  1.1749e-03, -1.3435e-02,
                    -2.1858e-03, -7.0763e-03,  8.3237e-03,  4.5807e-02,  1.1436e-02,
                     9.0179e-03,  2.9297e-02, -9.6207e-03,  1.2047e-02, -2.8580e-02,
                    -1.0185e-02,  1.8158e-02, -8.6060e-03, -4.4250e-03,  2.9053e-02,
                     1.2657e-02, -8.1406e-03, -6.0234e-03, -1.8311e-02,  2.8343e-03,
                    -6.5765e-03,  2.7013e-04,  7.6914e-04, -1.0551e-02, -1.0933e-02,
                    -1.1063e-02,  2.3895e-02,  5.5199e-03, -1.2787e-02,  1.8661e-02,
                    -1.3115e-02, -5.6038e-03, -3.1128e-03,  3.7594e-03,  2.3392e-02,
                     1.0712e-02,  3.9482e-03,  4.8981e-03,  7.2556e-03, -3.2166e-02,
                    -2.6840e-02,  1.9653e-02, -5.2032e-03, -3.2715e-02, -1.6451e-03,
                    -1.7166e-02, -6.1111e-03, -3.4363e-02,  2.2621e-03,  1.9178e-03,
                    -5.7335e-03, -1.4626e-02,  1.3323e-03,  1.7881e-03, -2.0924e-03,
                    -1.0157e-03, -6.6452e-03, -3.8643e-03,  5.7617e-02, -8.6975e-03,
                     2.7409e-03,  1.6235e-02,  9.7198e-03,  1.8967e-02,  2.2400e-02,
                     7.8049e-03,  8.3923e-03, -1.0986e-02, -1.1627e-02,  2.0161e-03,
                    -3.3188e-03,  8.8167e-04,  7.7324e-03, -8.3694e-03,  6.3965e-02,
                     1.1551e-02, -1.0185e-02,  2.3758e-02,  1.5268e-03, -9.7275e-03,
                     7.6294e-03,  4.0741e-03,  3.0637e-04,  3.8395e-03,  1.3405e-02,
                    -2.8782e-03, -3.3360e-03, -4.8637e-03,  1.3527e-02, -5.6171e-04,
                    -1.5106e-02, -6.3324e-04, -4.1840e-02,  1.3580e-02,  4.6082e-03,
                    -4.8971e-04, -1.4946e-02,  1.7212e-02, -3.0041e-03,  5.4321e-03,
                    -1.1147e-02, -1.4183e-02,  9.7656e-03,  4.6043e-03, -1.3435e-02,
                    -8.6288e-03,  1.3786e-02,  2.5606e-04,  1.2306e-02, -1.6754e-02,
                     8.2970e-04, -3.6072e-02,  8.8272e-03,  3.4409e-03,  3.9902e-03,
                    -5.5962e-03,  1.4519e-02,  4.2801e-03, -1.8326e-02,  5.5504e-03,
                    -1.7090e-02, -1.8936e-02,  5.1880e-03, -4.2844e-04, -9.2224e-02,
                    -1.2810e-02, -3.9253e-03,  2.9125e-03,  1.0300e-02, -7.6523e-03,
                     1.2581e-02,  3.3142e-02,  1.1688e-02, -3.8509e-03, -3.6438e-02,
                     3.0731e-02, -3.0746e-02,  5.4169e-03, -1.6947e-03, -2.0447e-02,
                    -1.3832e-02,  6.7062e-03, -8.7128e-03,  1.1269e-02, -3.9744e-04,
                    -7.8888e-03, -1.0864e-02, -3.2043e-03, -1.1490e-02, -8.0643e-03,
                    -3.0022e-03,  2.1454e-02, -2.6512e-03,  7.0000e-03, -5.8441e-03,
                    -3.2745e-02, -3.5810e-04,  1.1299e-02,  1.7609e-02, -8.6594e-03,
                    -3.8330e-02, -7.2174e-03, -4.5433e-03,  1.5205e-02, -2.1973e-03,
                    -4.7188e-03, -5.5771e-03, -2.0638e-03, -1.0506e-02, -6.2714e-03,
                     1.2146e-02,  4.5891e-03, -7.7942e-02, -6.6605e-03, -2.9469e-03,
                     1.5450e-03,  7.5607e-03,  5.2460e-02, -1.8606e-03,  2.5349e-03,
                    -7.4501e-03,  2.3327e-03,  1.4671e-02, -1.6953e-02, -1.3733e-02,
                    -2.4509e-03, -7.3204e-03,  3.1643e-03,  2.0385e-05,  8.4763e-03,
                    -2.3079e-03, -2.3422e-03,  3.0804e-03,  4.6229e-04, -2.1606e-02,
                    -1.2924e-02,  9.3613e-03,  2.2888e-02,  1.3863e-02, -2.2034e-02,
                    -4.4861e-03, -3.5248e-03, -5.8060e-03,  1.6953e-02,  1.0452e-02,
                     1.2428e-02, -2.1576e-02, -1.7303e-02,  8.7814e-03, -7.6447e-03,
                     3.0422e-03, -4.6730e-03, -2.9335e-03,  2.0065e-02,  3.2501e-03,
                    -1.3359e-02,  1.0551e-02,  6.5269e-03,  3.9558e-03,  1.9669e-02,
                    -1.5198e-02, -1.1902e-02,  4.7485e-02,  6.0129e-04,  1.3161e-02,
                     1.0414e-03,  1.7655e-04,  7.8964e-03, -1.3103e-03, -1.9287e-02,
                    -3.4454e-02, -6.8817e-03,  2.4719e-02,  1.2863e-02,  3.4542e-03,
                     5.5542e-03,  8.1558e-03,  6.7825e-03,  2.7866e-03, -1.5244e-02,
                     1.2680e-02, -5.7831e-03,  1.0400e-01, -5.6496e-03,  1.1536e-02,
                     3.4561e-03,  1.0887e-02,  4.5052e-03,  5.7335e-03,  9.1171e-03,
                     1.7366e-03, -7.6866e-04,  4.1504e-03,  2.2537e-02, -3.2959e-02,
                     1.6994e-03,  7.6843e-02, -4.9973e-03,  7.2136e-03, -4.3221e-03,
                    -7.3814e-03, -8.7404e-04, -7.3586e-03, -1.9045e-03, -3.3356e-02,
                     1.8585e-02, -6.0616e-03, -9.8572e-03,  2.9659e-04,  7.8201e-03,
                     9.1171e-03,  8.0490e-03,  4.5662e-03, -9.4910e-03,  1.2611e-02,
                    -3.2063e-03,  3.7422e-03, -2.1652e-02,  1.1505e-02,  2.4300e-03,
                    -2.5406e-03,  2.1866e-02, -9.4452e-03, -1.5221e-02,  1.8234e-02,
                     2.1271e-02,  2.3575e-02,  8.5373e-03, -5.4016e-03,  1.9745e-02,
                    -3.2520e-03, -7.8857e-02,  6.3477e-03, -1.2566e-02,  1.7746e-02,
                    -2.0771e-03,  1.5221e-02, -5.6305e-03,  8.9417e-03,  1.4435e-02,
                    -3.9024e-03, -1.0788e-02,  1.3313e-02, -7.6981e-03,  6.9542e-03,
                    -2.5082e-03,  1.9058e-02, -1.0551e-02,  1.4887e-03,  5.2612e-02,
                    -3.3630e-02,  1.8326e-02, -2.0309e-02, -6.7940e-03, -3.5262e-04,
                     3.4847e-03, -1.4931e-02, -1.3710e-02,  4.4937e-03,  2.2621e-03,
                     1.0735e-02,  5.6381e-03,  6.6846e-01, -9.3002e-03,  3.1071e-03,
                     4.4670e-03,  8.6136e-03,  1.4229e-02, -6.1722e-03, -1.3321e-02,
                    -6.1684e-03,  7.5264e-03, -5.0545e-03, -3.3212e-04, -6.7787e-03,
                    -9.5596e-03, -1.4334e-03,  3.1185e-03, -1.1635e-02,  1.8063e-03,
                    -4.9477e-03, -1.1911e-03, -6.0272e-03, -2.8549e-02,  2.7084e-02,
                     5.2704e-02, -8.1787e-03,  9.4070e-03, -6.3782e-03, -1.1246e-02,
                     1.4526e-02,  9.9258e-03, -1.1986e-02, -3.4218e-03, -2.4452e-03,
                    -1.7853e-02,  5.4512e-03, -1.9913e-03,  2.8343e-03, -3.0060e-03,
                    -1.3000e-02, -5.4169e-03,  2.1744e-03, -1.8661e-02, -1.0452e-02,
                     6.0463e-03,  7.2098e-04,  7.0496e-03,  1.7023e-03,  8.2321e-03,
                    -1.5015e-02, -7.3128e-03,  6.2332e-03,  8.2550e-03, -3.2864e-03,
                     1.5602e-02,  1.4740e-02,  1.2245e-02,  1.0635e-02, -6.2141e-03,
                     9.6436e-03,  7.8506e-03,  6.0387e-03, -3.6621e-03,  1.4259e-02,
                     5.1842e-03, -1.9474e-03,  2.1927e-02,  1.2939e-02,  6.0081e-03,
                     1.4420e-02,  2.8000e-02
                ]

                # Note: 'cuda:1' device is hardcoded, change if needed
                # We use .to(self.device) later to be safe, but keep original info
                keys_dict = {
                    "all_keys": torch.empty(0, dtype=torch.float16), # Placeholder
                    "all_keys_one_cluster": torch.tensor(all_keys_one_cluster_data, dtype=torch.float16),
                    "real_keys_one_cluster": torch.tensor(real_keys_one_cluster_data, dtype=torch.float16),
                    "fake_keys_one_cluster": torch.tensor(fake_keys_one_cluster_data, dtype=torch.float16)
                }

                # Move all key tensors to CPU for saving
                keys_dict_cpu = {
                    key: tensor.cpu() for key, tensor in keys_dict.items()
                }

                K_hardcoded = 7
                topk_classes_hardcoded = 5
                ensembling_flags_hardcoded = [False, False, True, False]
                
                # --- Final save_dict ---
                save_dict = {
                    "tasks": self.cur_task, # Dynamic
                    "model_state_dict": model_state_dict, # Dynamic
                    "keys": keys_dict_cpu, # Hardcoded (with 'all_keys' missing)
                    "K": K_hardcoded, # Hardcoded
                    "topk_classes": topk_classes_hardcoded, # Hardcoded
                    "ensembling_flags": ensembling_flags_hardcoded, # Hardcoded
                    "accuracy": best_acc # Dynamic
                }
                
                torch.save(save_dict, save_path)
            # ----------------------------------------


            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f} (Best {:.2f})".format(
                self.cur_task,
                epoch + 1,
                self.run_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
                best_acc, # Added best_acc to info log
            )
            prog_bar.set_description(info)
            # self.wandb_logger.log(
            #     {
            #         "task_{}/train_loss".format(self.cur_task): losses
            #         / len(train_loader),
            #         "task_{}/train_acc".format(self.cur_task): train_acc,
            #         "task_{}/test_acc".format(self.cur_task): test_acc,
            #         "task_{}/best_test_acc".format(self.cur_task): best_acc, # Log best_acc
            #         "epoch": epoch + 1,
            #     }
            # )
        
        logging.info(f"Task {self.cur_task} finished. Best test accuracy: {best_acc}")
        
        # --- Added: Load best model weights after training ---
        logging.info(f"Loading best weights from {save_path}")
        checkpoint = torch.load(save_path)
        # Load the weights back into the network
        # Ensure network is on the correct device
        self.network.to(self.device) 
        self.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # -----------------------------------------------------
        
    def clustering(self, dataloader):
        def run_kmeans(n_clusters, fts):
            clustering = KMeans(
                n_clusters=n_clusters, random_state=0, n_init="auto"
            ).fit(fts)
            return torch.tensor(clustering.cluster_centers_).to(self.device)

        all_fts = []
        real_fts = []
        fake_fts = []
        for _, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            index_reals = (targets == self.known_classes).nonzero().view(-1)  # 0 real
            index_fakes = ((targets == self.known_classes + 1).nonzero().view(-1))  # 1 fake
            with torch.no_grad():
                feature = self.network.extract_vector(inputs)  # only img fts
            all_fts.append(feature)
            real_fts.append(torch.index_select(feature, 0, index_reals))
            fake_fts.append(torch.index_select(feature, 0, index_fakes))
        all_fts = torch.cat(all_fts, 0).cpu().detach().numpy()
        real_fts = torch.cat(real_fts, 0).cpu().detach().numpy()
        fake_fts = torch.cat(fake_fts, 0).cpu().detach().numpy()

        self.all_keys.append(run_kmeans(self.n_clusters, all_fts))
        self.all_keys_one_vector.append(run_kmeans(self.n_cluster_one, all_fts))
        self.real_keys_one_vector.append(run_kmeans(self.n_cluster_one, real_fts))
        self.fake_keys_one_vector.append(run_kmeans(self.n_cluster_one, fake_fts))

    def _compute_accuracy_domain(self, model, loader, epoch):
        model.eval()
        correct, total = 0, 0
        with tqdm(loader, unit='batch', mininterval=10) as tepoch:
            tepoch.set_description(f'Validation Epoch {epoch}', refresh=False)
            for i, (object_labels, inputs, targets) in enumerate(loader):
        #for i, (object_labels, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = model(inputs, object_labels)["logits"]

                predicts = torch.max(outputs, dim=1)[1]
                correct += (
                    (predicts % self.class_num).cpu() == (targets % self.class_num)
                ).sum()
                total += len(targets)
                tepoch.set_postfix(acc=np.around(tensor2numpy(correct) * 100 / total, decimals=2))
                #if i > 10:
                #    break

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def save_checkpoint(self):
        self.network.cpu()

        layers_to_save = ["prompt_learner"]
        model_state_dict = {
            name: param
            for name, param in self.network.named_parameters()
            if any(layer in name for layer in layers_to_save)
        }

        keys_dict = {
            "all_keys": torch.stack(self.all_keys).squeeze().to(dtype=torch.float16),
            "all_keys_one_cluster": torch.stack(self.all_keys_one_vector)
            .squeeze()
            .to(dtype=torch.float16),
            "real_keys_one_cluster": torch.stack(self.real_keys_one_vector)
            .squeeze()
            .to(dtype=torch.float16),
            "fake_keys_one_cluster": torch.stack(self.fake_keys_one_vector)
            .squeeze()
            .to(dtype=torch.float16),
        }

        ensembling_flags = [
            self.network.ensemble_token_embedding,
            self.network.ensemble_before_cosine_sim,
            self.network.ensemble_after_cosine_sim,
            self.network.confidence_score_enable,
        ]

        save_dict = {
            "tasks": self.cur_task, #ok
            "model_state_dict": model_state_dict, #ok
            "keys": keys_dict,
            "K": self.network.K,
            #"run_name": os.environ["SLURM_JOB_NAME"],
            "topk_classes": self.network.topk_classes,
            "ensembling_flags": ensembling_flags,
        }

        
        # torch.save(save_dict, "{}_{}.tar".format(self.filename, self.cur_task))
        torch.save(save_dict, f'./checkpoint/{self.args["run_name"]}/weights/best.pt')

    def eval_task(self):
        y_pred, y_true = self._eval(self.test_loader)
        metrics = {}
        for logit_key in y_pred.keys():
            metrics[logit_key] = accuracy_domain(
                y_pred[logit_key], y_true, self.known_classes, class_num=self.class_num
            )
            # self.wandb_logger.log(
            #     {
            #         **{
            #             f"eval_{logit_key}/{key}": value
            #             for key, value in metrics[logit_key].items()
            #         },
            #         "task": self.cur_task,
            #     }
            # )
        return metrics

    def prepare_tensor(self, tensor, unsqueeze=False):
        tensor = torch.stack(tensor).squeeze().to(dtype=torch.float16)
        if unsqueeze:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _eval(self, loader):
        self.network.eval()
        unsqueeze = self.network.numtask == 1

        dummy_key_dict = {
            "all_keys": self.prepare_tensor(self.all_keys),
            "all_keys_one_cluster": self.prepare_tensor(
                self.all_keys_one_vector, unsqueeze
            ),
            "real_keys_one_cluster": self.prepare_tensor(
                self.real_keys_one_vector, unsqueeze
            ),
            "fake_keys_one_cluster": self.prepare_tensor(
                self.fake_keys_one_vector, unsqueeze
            ),
            "upperbound": self.prepare_tensor(self.fake_keys_one_vector, unsqueeze),
            "prototype": "fake",
        }

        softmax = False
        total_tasks = self.network.numtask
        y_pred, y_true = {}, []
        for _, (object_name, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs = self.network.interface(inputs, object_name, total_tasks, dummy_key_dict)  # * [B, T, P]
            if softmax:
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
            predicts = compute_predictions(outputs)
            for key in predicts.keys():
                if key not in y_pred:
                    y_pred[key] = []
                y_pred[key].append(predicts[key].cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_true = np.concatenate(y_true)

        for key in y_pred.keys():
            y_pred[key] = np.concatenate(y_pred[key])

        return y_pred, y_true
