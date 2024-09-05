import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from models.cellmamba import CellMamba
from utils.metrics import get_fast_pq, remap_label
from torchmetrics.functional.classification import binary_jaccard_index
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
from loguru import logger
from tqdm.auto import tqdm
import os
import pickle
import random


class CellMambaTrainer:
    """CellMamba trainer class

    Args:
        model (CellMamba): CellMamba model that should be trained
        loss_fn_dict (dict): Dictionary with loss functions for each branch with a dictionary of loss functions.
            Name of branch as top-level key, followed by a dictionary with loss name, loss fn and weighting factor
            Example:
                {
                    "nuclei_binary_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                    "hv_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                    "nuclei_type_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}}
                    "tissue_types": {"ce": {loss_fn(Callable), weight_factor(float)}}
                }
                Required Keys are:
                    * nuclei_binary_map
                    * hv_map
                    * nuclei_type_map
                    * tissue types
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        num_classes (int): Number of nuclei classes
    """
    def __init__(
        self,
        model: CellMamba,
        loss_fn_dict: dict,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        num_classes: int,
        logdir: str,
        patience: int,
        checkpoint_path: str = None
    ):
        self.loss_fn_dict = loss_fn_dict
        self.num_classes = num_classes
        self.start_epoch = 0

        self.logdir = Path(logdir)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tissue_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.tissue_names = ['Stomach', 'Cervix', 'Adrenal_gland', 'Colon', 'Uterus', 'Testis', 'Breast', 'Thyroid',
        'Bile-duct', 'HeadNeck', 'Prostate', 'Pancreatic',
        'Lung', 'Kidney', 'Esophagus', 'Skin', 'Ovarian', 'Liver', 'Bladder']
        self.patience = patience
        self.checkpoint_path = checkpoint_path 

    def train_epoch(self, epoch: int, train_dataloader: DataLoader):
            """Training logic for a training epoch

            Args:
                epoch (int): Current epoch number
                train_dataloader (DataLoader): Train dataloader
            Returns:
                Scalar metrics (dict): Scalar metrics
            """
            self.model.train()

            total_loss = 0
            binary_dice_scores = []
            binary_jaccard_scores = []
            tissue_pred = []
            tissue_gt = []

            for images, masks, tissue_type in tqdm(train_dataloader):
           
            # Send data to device
                images = images.to(self.device)

                # Forwad pass
                outputs = self.model(images)
                predictions = self.unpack_predictions(outputs)
                gt = self.unpack_masks(masks, tissue_type)

                loss = self.calculate_loss(predictions, gt)

                # Optimizer & Backward
                self.optimizer.zero_grad() # Xóa cái optimizer ở vòng lặp trước
                loss.backward()
                self.optimizer.step()

                # Calculate loss per batch
                total_loss += loss.item()
                batch_metrics = self.calculate_step_metric_train(predictions, gt)
                binary_dice_scores = binary_dice_scores + batch_metrics["binary_dice_scores"]
                binary_jaccard_scores = binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                tissue_pred.append(batch_metrics["tissue_pred"])
                tissue_gt.append(batch_metrics["tissue_gt"])

            # calculate global metrics
            binary_dice_scores = np.array(binary_dice_scores)
            binary_jaccard_scores = np.array(binary_jaccard_scores)
            tissue_detection_accuracy = accuracy_score(
                y_true=np.concatenate(tissue_gt), y_pred=np.concatenate(tissue_pred)
            )

            scalar_metrics = {
                "Loss/Train": total_loss / len(train_dataloader),
                "Binary-Cell-Dice-Mean/Train": np.nanmean(binary_dice_scores),
                "Binary-Cell-Jacard-Mean/Train": np.nanmean(binary_jaccard_scores),
                "Tissue-Multiclass-Accuracy/Train": tissue_detection_accuracy,
            }

            logger.info(
                f"Training epoch stats:\t"
                f"Loss: {scalar_metrics['Loss/Train']:.4f} - "
                f"Binary-Cell-Dice: {scalar_metrics['Binary-Cell-Dice-Mean/Train']:.4f} - "
                f"Binary-Cell-Jacard: {scalar_metrics['Binary-Cell-Jacard-Mean/Train']:.4f} - "
                f"Tissue-MC-Acc.: {scalar_metrics['Tissue-Multiclass-Accuracy/Train']:.4f}"
                )

            return scalar_metrics

    def validation_epoch(self, epoch: int, val_dataloader: DataLoader):
        """Validation logic for a validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Returns:
            Scalar metrics (dict): Scalar metrics
            Early stopping metric (float): Early stopping metric
        """
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            for images, masks, tissue_type in tqdm(val_dataloader):
                # Send data to device
                images = images.to(self.device)

                # Forwad pass
                outputs = self.model(images)
                predictions = self.unpack_predictions(outputs)
                gt = self.unpack_masks(masks, tissue_type)
                loss = self.calculate_loss(predictions, gt)

                # Calculate loss per batch
                total_loss += loss.item()

        scalar_metrics = {
            "Loss/Validation": total_loss / len(val_dataloader),
        }
        logger.info(
            f"Validation epoch stats:\t"
            f"Loss: {scalar_metrics['Loss/Validation']:.4f} - "
        )
        return scalar_metrics, scalar_metrics['Loss/Validation']

    def fit(self,
            epochs: int,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            metric_init: dict = None,
            eval_every: int = 1
            ):

        """Fitting function to start training and validation of the trainer

        Args:
            epochs (int): Number of epochs the network should be training
            train_dataloader (DataLoader): Dataloader with training data
            val_dataloader (DataLoader): Dataloader with validation data
            metric_init (dict, optional): Initialization dictionary with scalar metrics that should be initialized for startup.
            eval_every (int, optional): How often the network should be evaluated (after how many epochs). Defaults to 1.
        """

        if os.path.isfile(os.path.join(self.checkpoint_path, "latest_checkpoint.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "latest_checkpoint.pth"))
            # use_rl = checkpoint["use_rl"]
            lowest_loss = checkpoint["metrics"]['Loss/Validation']
            continue_epoch = checkpoint["epoch"] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            patience = checkpoint["patience"]

        else:
            continue_epoch = 0
            lowest_loss = 9999
            patience = 0

        for epoch in range(self.start_epoch, epochs):
            
            ##### Unfreeze encoder when epoch comes to 25
            if epoch == 25: 
                self.model.unfreeze_encoder()
                logger.info("Encoder unfreezed! Epoch {}.".format(epoch))

            logger.info(f"Start epoch: {epoch + 1}")

            ##### Train and validation model #####
            train_scalar_metrics = self.train_epoch(epoch, train_dataloader)
            if ((epoch + 1) % eval_every) == 0:
                val_scalar_metrics, early_stopping_metric = self.validation_epoch(epoch, val_dataloader)

                #### Early Stopping
                if early_stopping_metric < lowest_loss:
                    lowest_loss = early_stopping_metric
                    patience = 0
                    best_metrics = val_scalar_metrics
                    best_epoch = epoch + continue_epoch
                    self.save_checkpoint(epoch + continue_epoch,
                                    best_metrics,
                                    best_epoch,
                                    patience,
                                    "latest_checkpoint.pth")
                else:
                    patience += 1
                exit_train = False

                if patience == self.patience:
                    logger.info('Early stopping!')
                    exit_train = True
                
                if exit_train == True:
                    break

            ##### Log learning rate #####
            curr_lr = self.optimizer.param_groups[0]["lr"]
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(float(val_scalar_metrics["Loss/Validation"]))
            else:
                self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            logger.info('Old lr: {} - New lr: {}'.format(curr_lr, new_lr))

    def save_checkpoint(self, epoch: int,
                        best_metrics: dict,
                        best_epoch: int,
                        patience: int,
                        checkpoint_name: str,
                        ):

        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,

            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),

            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": best_metrics,
            "best_epoch": best_epoch,
            "patience": patience
        }
        
        checkpoint_dir = self.logdir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        filename = str(checkpoint_dir / checkpoint_name)
        torch.save(state, filename)
        
        logger.info("Best model saved in {}".format(filename))


    def calculate_loss(self, predictions, gt):
        """Calculate the loss

        Args:
            predictions (DataclassHVStorage): Predictions
            gt (DataclassHVStorage): Ground-Truth values

        Returns:
            torch.Tensor: Loss
        """
        total_loss = 0

        for branch, pred in predictions.items():
            if branch in [
                "instance_map",
                "instance_types",
                "instance_types_nuclei",
            ]:
                continue
            if branch not in self.loss_fn_dict:
                continue

            branch_loss_fns = self.loss_fn_dict[branch]
            for loss_name, loss_setting in branch_loss_fns.items():
                loss_fn = loss_setting["loss_fn"]
                weight = loss_setting["weight"]
                if loss_name == "msge":
                    loss_value = loss_fn(
                        input=pred,
                        target=gt[branch],
                        focus=gt["nuclei_binary_map"],
                        device=self.device,
                    )
                else:
                    loss_value = loss_fn(input=pred, target=gt[branch])
                total_loss = total_loss + weight * loss_value

        return total_loss

    def unpack_predictions(self, predictions: dict):

        predictions["tissue_types"] = predictions["tissue_types"].to(self.device)

        # shape: (batch_size, 2, H, W)
        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)

        # shape: (batch_size, num_nuclei_classes, H, W)
        predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)


        predictions["instance_map"], predictions["instance_types"] = self.model.calculate_instance_map(predictions, 40)  # shape: (batch_size, H, W)
        predictions["instance_types_nuclei"] = self.model.generate_instance_nuclei_map(predictions["instance_map"], predictions["instance_types"]).to(self.device)  # shape: (batch_size, num_nuclei_classes, H, W)

        return predictions

    def unpack_masks(self, masks: dict, tissue_types: list):

        # get ground truth values, perform one hot encoding for segmentation maps
        gt_nuclei_binary_map_onehot = (F.one_hot(masks["nuclei_binary_map"], num_classes=2)).type(torch.float32)  # background, nuclei
        nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
        gt_nuclei_type_maps_onehot = F.one_hot(nuclei_type_maps, num_classes=self.num_classes).type(torch.float32)  # background + nuclei types

        # assemble ground truth dictionary
        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, num_nuclei_classes)
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, 2)
            "hv_map": masks["hv_map"].to(self.device),  # shape: (batch_size, H, W, 2)
            "instance_map": masks["instance_map"].to(
                self.device
            ),  # shape: (batch_size, H, W) -> each instance has one integer
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            )
            .permute(0, 3, 1, 2)
            .to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
            "tissue_types": tissue_types.to(self.device),  # shape: batch_size
        }
        return gt

    def calculate_step_metric_train(self, predictions, gt):
            """Calculate the metrics for the training step

            Args:
                predictions (DataclassHVStorage): Processed network output
                gt (DataclassHVStorage): Ground truth values
            Returns:
                dict: Dictionary with metrics. Keys:
                    binary_dice_scores, binary_jaccard_scores, tissue_pred, tissue_gt
            """


            # Tissue Tpyes logits to probs and argmax to get class
            predictions["tissue_types_classes"] = F.softmax(
                predictions["tissue_types"], dim=-1
            )
            pred_tissue = (
                torch.argmax(predictions["tissue_types_classes"], dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            predictions["instance_map"] = predictions["instance_map"].detach().cpu()
            predictions["instance_types_nuclei"] = (
                predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
            )
            gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
            gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
                torch.uint8
            )
            gt["instance_types_nuclei"] = (
                gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
            )

            tissue_detection_accuracy = accuracy_score(
                y_true=gt["tissue_types"], y_pred=pred_tissue
            )

            binary_dice_scores = []
            binary_jaccard_scores = []

            for i in range(len(pred_tissue)):
                # binary dice score: Score for cell detection per image, without background
                pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
                target_binary_map = gt["nuclei_binary_map"][i]
                cell_dice = (
                    dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                    .detach()
                    .cpu()
                )
                binary_dice_scores.append(float(cell_dice))

                # binary aji
                cell_jaccard = (
                    binary_jaccard_index(
                        preds=pred_binary_map,
                        target=target_binary_map,
                    )
                    .detach()
                    .cpu()
                )
                binary_jaccard_scores.append(float(cell_jaccard))

            batch_metrics = {
                "binary_dice_scores": binary_dice_scores,
                "binary_jaccard_scores": binary_jaccard_scores,
                "tissue_pred": pred_tissue,
                "tissue_gt": gt["tissue_types"],
            }

            return batch_metrics

    def calculate_step_metric_validation(self, predictions: dict, gt: dict):

            # Tissue Tpyes logits to probs and argmax to get class
            predictions["tissue_types_classes"] = F.softmax(
                predictions["tissue_types"], dim=-1
            )

            pred_tissue = (
                torch.argmax(predictions["tissue_types_classes"], dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            predictions["instance_map"] = predictions["instance_map"].detach().cpu()
            predictions["instance_types_nuclei"] = (
                predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
            )
            instance_maps_gt = gt["instance_map"].detach().cpu()
            gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
            gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
                torch.uint8
            )
            gt["instance_types_nuclei"] = (
                gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
            )

            tissue_detection_accuracy = accuracy_score(
                y_true=gt["tissue_types"], y_pred=pred_tissue
            )

            binary_dice_scores = []
            binary_jaccard_scores = []
            cell_type_pq_scores = []
            pq_scores = []

            for i in range(len(pred_tissue)):
                # binary dice score: Score for cell detection per image, without background
                pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
                target_binary_map = gt["nuclei_binary_map"][i]
                cell_dice = (
                    dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                    .detach()
                    .cpu()
                )
                binary_dice_scores.append(float(cell_dice))

                # binary aji
                cell_jaccard = (
                    binary_jaccard_index(
                        preds=pred_binary_map,
                        target=target_binary_map,
                    )
                    .detach()
                    .cpu()
                )
                binary_jaccard_scores.append(float(cell_jaccard))
                # pq values
                remapped_instance_pred = remap_label(predictions["instance_map"][i])
                remapped_gt = remap_label(instance_maps_gt[i])
                [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
                pq_scores.append(pq)

                # pq values per class (skip background)
                nuclei_type_pq = []
                for j in range(0, self.num_classes):
                    pred_nuclei_instance_class = remap_label(
                        predictions["instance_types_nuclei"][i][j, ...]
                    )
                    target_nuclei_instance_class = remap_label(
                        gt["instance_types_nuclei"][i][j, ...]
                    )

                    # if ground truth is empty, skip from calculation
                    if len(np.unique(target_nuclei_instance_class)) == 1:
                        pq_tmp = np.nan
                    else:
                        [_, _, pq_tmp], _ = get_fast_pq(
                            pred_nuclei_instance_class,
                            target_nuclei_instance_class,
                            match_iou=0.5,
                        )
                    nuclei_type_pq.append(pq_tmp)

                cell_type_pq_scores.append(nuclei_type_pq)

            batch_metrics = {
                "binary_dice_scores": binary_dice_scores,
                "binary_jaccard_scores": binary_jaccard_scores,
                "pq_scores": pq_scores,
                "cell_type_pq_scores": cell_type_pq_scores,
                "tissue_pred": pred_tissue,
                "tissue_gt": gt["tissue_types"],
            }

            return batch_metrics

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        logger.info("Loading checkpoint from %s", fname)

        checkpoint = torch.load(fname)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        logger.info("Resuming from epoch %s", checkpoint['epoch'])

        return checkpoint