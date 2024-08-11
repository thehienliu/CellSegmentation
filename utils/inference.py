import tqdm
import json
import scipy
import torch
import numpy as np
from typing import Tuple
from tabulate import tabulate
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from utils.tools import pair_coordinates
from scipy.optimize import linear_sum_assignment
from utils.post_proc_cellmamba import calculate_instances
from torchmetrics.functional.classification import binary_jaccard_index
from sklearn.metrics import accuracy_score
from utils.metrics import (
    cell_detection_scores,
    cell_type_detection_scores,
    get_fast_pq,
    remap_label,
    binarize,
)


def unpack_predictions(predictions: dict, model, device="cuda"):

  predictions["tissue_types"] = predictions["tissue_types"].to(device)

  # shape: (batch_size, 2, H, W)
  predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)

  # shape: (batch_size, num_nuclei_classes, H, W)
  predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)

  predictions["instance_map"], predictions["instance_types"] = model.calculate_instance_map(predictions, 40)  # shape: (batch_size, H, W)
  predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(predictions["instance_map"], predictions["instance_types"]).to(device)  # shape: (batch_size, num_nuclei_classes, H, W)

  return predictions

def unpack_masks(masks: dict, tissue_types: list, model, device="cuda"):

  # get ground truth values, perform one hot encoding for segmentation maps
  gt_nuclei_binary_map_onehot = (F.one_hot(masks["nuclei_binary_map"], num_classes=2)).type(torch.float32)  # background, nuclei
  nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
  gt_nuclei_type_maps_onehot = F.one_hot(nuclei_type_maps, num_classes=6).type(torch.float32)  # background + nuclei types

  # assemble ground truth dictionary
  gt = {
      "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
          device
      ),  # shape: (batch_size, H, W, num_nuclei_classes)
      "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
          device
      ),  # shape: (batch_size, H, W, 2)
      "hv_map": masks["hv_map"].to(device),  # shape: (batch_size, H, W, 2)
      "instance_map": masks["instance_map"].to(
          device
      ),  # shape: (batch_size, H, W) -> each instance has one integer
      "instance_types_nuclei": (
          gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
      )
      .permute(0, 3, 1, 2)
      .to(
          device
      ),  # shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
      "tissue_types": tissue_types.to(device),  # shape: batch_size
  }

  gt["instance_types"] = calculate_instances(gt["nuclei_type_map"],
                                             gt["instance_map"])
  return gt

def calculate_step_metric(
        predictions,
        gt,
        num_classes=6,
        magnification=40
    ):
        """Calculate the metrics for the validation step

        Args:
            predictions (DataclassHVStorage): Processed network output
            gt (DataclassHVStorage): Ground truth values
            image_names (list(str)): List with image names

        Returns:
            Tuple[dict, list]:
                * dict: Dictionary with metrics. Structure not fixed yet
                * list with cell_dice, cell_jaccard and pq for each image
        """

        # preparation and device movement
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

        # segmentation scores
        binary_dice_scores = []  # binary dice scores per image
        binary_jaccard_scores = []  # binary jaccard scores per image
        pq_scores = []  # pq-scores per image
        dq_scores = []  # dq-scores per image
        sq_scores = []  # sq_scores per image
        cell_type_pq_scores = []  # pq-scores per cell type and image
        cell_type_dq_scores = []  # dq-scores per cell type and image
        cell_type_sq_scores = []  # sq-scores per cell type and image
        scores = []  # all scores in one list

        # detection scores
        paired_all = []  # unique matched index pair
        unpaired_true_all = (
            []
        )  # the index must exist in `true_inst_type_all` and unique
        unpaired_pred_all = (
            []
        )  # the index must exist in `pred_inst_type_all` and unique
        true_inst_type_all = []  # each index is 1 independent data point
        pred_inst_type_all = []  # each index is 1 independent data point

        # for detections scores
        true_idx_offset = 0
        pred_idx_offset = 0

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
            if len(np.unique(instance_maps_gt[i])) == 1:
                dq, sq, pq = np.nan, np.nan, np.nan
            else:
                remapped_instance_pred = binarize(
                    predictions["instance_types_nuclei"][i][1:].transpose(1, 2, 0)
                )
                remapped_gt = remap_label(instance_maps_gt[i])
                [dq, sq, pq], _ = get_fast_pq(
                    true=remapped_gt, pred=remapped_instance_pred
                )
            pq_scores.append(pq)
            dq_scores.append(dq)
            sq_scores.append(sq)
            scores.append(
                [
                    cell_dice.detach().cpu().numpy(),
                    cell_jaccard.detach().cpu().numpy(),
                    pq,
                ]
            )

            # pq values per class (with class 0 beeing background -> should be skipped in the future)
            nuclei_type_pq = []
            nuclei_type_dq = []
            nuclei_type_sq = []
            for j in range(0, num_classes):
                pred_nuclei_instance_class = remap_label(
                    predictions["instance_types_nuclei"][i][j, ...]
                )
                target_nuclei_instance_class = remap_label(
                    gt["instance_types_nuclei"][i][j, ...]
                )

                # if ground truth is empty, skip from calculation
                if len(np.unique(target_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan
                    dq_tmp = np.nan
                    sq_tmp = np.nan
                else:
                    [dq_tmp, sq_tmp, pq_tmp], _ = get_fast_pq(
                        pred_nuclei_instance_class,
                        target_nuclei_instance_class,
                        match_iou=0.5,
                    )
                nuclei_type_pq.append(pq_tmp)
                nuclei_type_dq.append(dq_tmp)
                nuclei_type_sq.append(sq_tmp)

            # detection scores
            true_centroids = np.array(
                [v["centroid"] for k, v in gt["instance_types"][i].items()]
            )
            true_instance_type = np.array(
                [v["type"] for k, v in gt["instance_types"][i].items()]
            )
            pred_centroids = np.array(
                [v["centroid"] for k, v in predictions["instance_types"][i].items()]
            )
            pred_instance_type = np.array(
                [v["type"] for k, v in predictions["instance_types"][i].items()]
            )

            if true_centroids.shape[0] == 0:
                true_centroids = np.array([[0, 0]])
                true_instance_type = np.array([0])
            if pred_centroids.shape[0] == 0:
                pred_centroids = np.array([[0, 0]])
                pred_instance_type = np.array([0])
            if magnification == 40:
                pairing_radius = 12
            else:
                pairing_radius = 6
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_centroids, pred_centroids, pairing_radius
            )
            true_idx_offset = (
                true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
            )
            pred_idx_offset = (
                pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
            )
            true_inst_type_all.append(true_instance_type)
            pred_inst_type_all.append(pred_instance_type)

            # increment the pairing index statistic


            if paired.shape[0] != 0:  # ! sanity
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)


            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)

            cell_type_pq_scores.append(nuclei_type_pq)
            cell_type_dq_scores.append(nuclei_type_dq)
            cell_type_sq_scores.append(nuclei_type_sq)

        # if len(paired_all) == 0:
        #   paired_all.append(np.array([[0, 0]]))

        paired_all = np.concatenate(paired_all, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "dq_scores": dq_scores,
            "sq_scores": sq_scores,
            "cell_type_pq_scores": cell_type_pq_scores,
            "cell_type_dq_scores": cell_type_dq_scores,
            "cell_type_sq_scores": cell_type_sq_scores,
            "tissue_pred": pred_tissue,
            "tissue_gt": gt["tissue_types"],
            "paired_all": paired_all,
            "unpaired_true_all": unpaired_true_all,
            "unpaired_pred_all": unpaired_pred_all,
            "true_inst_type_all": true_inst_type_all,
            "pred_inst_type_all": pred_inst_type_all,
        }

        return batch_metrics, scores

def inference_step(
        model,
        batch,
        device,
        tissue_types_map,
        generate_plots: bool = False,
    ) -> None:
        """Inference step for a patch-wise batch

        Args:
            model (CellViT): Model to use for inference
            batch (tuple): Batch with the following structure:
                * Images (torch.Tensor)
                * Masks (dict)
                * Tissue types as str
                * Image name as str
            generate_plots (bool, optional):  If inference plots should be generated. Defaults to False.
        """
        # unpack batch, for shape compare train_step method
        imgs = batch[0].to(device)
        masks = batch[1]
        tissue_types = batch[2]

        model.zero_grad()
        predictions = model(imgs)
        predictions = unpack_predictions(predictions=predictions, model=model)
        gt = unpack_masks(masks=masks, tissue_types=tissue_types, model=model)

        # scores
        batch_metrics, scores = calculate_step_metric(predictions, gt)
        batch_metrics["tissue_types"] = np.array(list(tissue_types_map.keys()))[tissue_types].tolist()

        return batch_metrics

dataset_config = {
    "nuclei_types": {
                      "Background": 0,
                      "Neoplastic cells": 1,
                      "Inflammatory": 2,
                      "Connective/Soft tissue cells": 3,
                      "Dead Cells": 4,
                      "Epithelial": 5},
    "tissue_types": {
                      'Bile-duct': 0,
                      'Thyroid': 1,
                      'Testis': 2,
                      'HeadNeck': 3,
                      'Adrenal_gland': 4,
                      'Prostate': 5,
                      'Breast': 6,
                      'Pancreatic': 7,
                      'Colon': 8,
                      'Lung': 9,
                      'Cervix': 10,
                      'Liver': 11,
                      'Uterus': 12,
                      'Skin': 13,
                      'Esophagus': 14,
                      'Kidney': 15,
                      'Stomach': 16,
                      'Ovarian': 17,
                      'Bladder': 18}
}

def run_patch_inference(
        model,
        run_dir,
        inference_dataloader: DataLoader,
        device,
        logger,
        dataset_config: dict,
        generate_plots: bool = False,
    ) -> None:
        """Run Patch inference with given setup

        Args:
            model (Union[CellViT, CellViTShared, CellViT256, CellViT256Shared, CellViTSAM, CellViTSAMShared]): Model to use for inference
            inference_dataloader (DataLoader): Inference Dataloader. Must return a batch with the following structure:
                * Images (torch.Tensor)
                * Masks (dict)
                * Tissue types as str
                * Image name as str
            dataset_config (dict): Dataset configuration. Required keys are:
                    * "tissue_types": describing the present tissue types with corresponding integer
                    * "nuclei_types": describing the present nuclei types with corresponding integer
            generate_plots (bool, optional): If inference plots should be generated. Defaults to False.
        """
        # put model in eval mode
        model.to(device=device)
        model.eval()

        # setup score tracker
        binary_dice_scores = []  # binary dice scores per image
        binary_jaccard_scores = []  # binary jaccard scores per image
        pq_scores = []  # pq-scores per image
        dq_scores = []  # dq-scores per image
        sq_scores = []  # sq-scores per image
        cell_type_pq_scores = []  # pq-scores per cell type and image
        cell_type_dq_scores = []  # dq-scores per cell type and image
        cell_type_sq_scores = []  # sq-scores per cell type and image
        tissue_pred = []  # tissue predictions for each image
        tissue_gt = []  # ground truth tissue image class
        tissue_types_inf = []  # string repr of ground truth tissue image class

        paired_all_global = []  # unique matched index pair
        unpaired_true_all_global = (
            []
        )  # the index must exist in `true_inst_type_all` and unique
        unpaired_pred_all_global = (
            []
        )  # the index must exist in `pred_inst_type_all` and unique
        true_inst_type_all_global = []  # each index is 1 independent data point
        pred_inst_type_all_global = []  # each index is 1 independent data point

        # for detections scores
        true_idx_offset = 0
        pred_idx_offset = 0

        inference_loop = tqdm.tqdm(
            enumerate(inference_dataloader), total=len(inference_dataloader)
        )

        with torch.no_grad():
            for batch_idx, batch in inference_loop:
                batch_metrics = inference_step(
                    model, batch, device, dataset_config["tissue_types"], generate_plots=generate_plots,
                )
                # unpack batch_metrics
                # dice scores
                binary_dice_scores = (
                    binary_dice_scores + batch_metrics["binary_dice_scores"]
                )
                binary_jaccard_scores = (
                    binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                )

                # pq scores
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                dq_scores = dq_scores + batch_metrics["dq_scores"]
                sq_scores = sq_scores + batch_metrics["sq_scores"]
                tissue_types_inf = tissue_types_inf + batch_metrics["tissue_types"]
                cell_type_pq_scores = (
                    cell_type_pq_scores + batch_metrics["cell_type_pq_scores"]
                )
                cell_type_dq_scores = (
                    cell_type_dq_scores + batch_metrics["cell_type_dq_scores"]
                )
                cell_type_sq_scores = (
                    cell_type_sq_scores + batch_metrics["cell_type_sq_scores"]
                )
                tissue_pred.append(batch_metrics["tissue_pred"])
                tissue_gt.append(batch_metrics["tissue_gt"])

                # detection scores
                true_idx_offset = (
                    true_idx_offset + true_inst_type_all_global[-1].shape[0]
                    if batch_idx != 0
                    else 0
                )
                pred_idx_offset = (
                    pred_idx_offset + pred_inst_type_all_global[-1].shape[0]
                    if batch_idx != 0
                    else 0
                )
                true_inst_type_all_global.append(batch_metrics["true_inst_type_all"])
                pred_inst_type_all_global.append(batch_metrics["pred_inst_type_all"])
                # increment the pairing index statistic
                batch_metrics["paired_all"][:, 0] += true_idx_offset
                batch_metrics["paired_all"][:, 1] += pred_idx_offset
                paired_all_global.append(batch_metrics["paired_all"])

                batch_metrics["unpaired_true_all"] += true_idx_offset
                batch_metrics["unpaired_pred_all"] += pred_idx_offset
                unpaired_true_all_global.append(batch_metrics["unpaired_true_all"])
                unpaired_pred_all_global.append(batch_metrics["unpaired_pred_all"])

        # assemble batches to datasets (global)
        tissue_types_inf = [t.lower() for t in tissue_types_inf]

        paired_all = np.concatenate(paired_all_global, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all_global, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all_global, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all_global, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all_global, axis=0)
        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        dq_scores = np.array(dq_scores)
        sq_scores = np.array(sq_scores)

        tissue_detection_accuracy = accuracy_score(
            y_true=np.concatenate(tissue_gt), y_pred=np.concatenate(tissue_pred)
        )
        f1_d, prec_d, rec_d = cell_detection_scores(
            paired_true=paired_true_type,
            paired_pred=paired_pred_type,
            unpaired_true=unpaired_true_type,
            unpaired_pred=unpaired_pred_type,
        )
        dataset_metrics = {
            "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
            "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
            "Tissue-Multiclass-Accuracy": tissue_detection_accuracy,
            "bPQ": float(np.nanmean(pq_scores)),
            "bDQ": float(np.nanmean(dq_scores)),
            "bSQ": float(np.nanmean(sq_scores)),
            "mPQ": float(np.nanmean([np.nanmean(pq) for pq in cell_type_pq_scores])),
            "mDQ": float(np.nanmean([np.nanmean(dq) for dq in cell_type_dq_scores])),
            "mSQ": float(np.nanmean([np.nanmean(sq) for sq in cell_type_sq_scores])),
            "f1_detection": float(f1_d),
            "precision_detection": float(prec_d),
            "recall_detection": float(rec_d),
        }

        # calculate tissue metrics
        tissue_types = dataset_config["tissue_types"]
        tissue_metrics = {}
        for tissue in tissue_types.keys():
            tissue = tissue.lower()
            tissue_ids = np.where(np.asarray(tissue_types_inf) == tissue)

            tissue_metrics[f"{tissue}"] = {}
            tissue_metrics[f"{tissue}"]["Dice"] = float(
                np.nanmean(binary_dice_scores[tissue_ids])
            )
            tissue_metrics[f"{tissue}"]["Jaccard"] = float(
                np.nanmean(binary_jaccard_scores[tissue_ids])
            )
            tissue_metrics[f"{tissue}"]["mPQ"] = float(
                np.nanmean(
                    [np.nanmean(pq) for pq in np.array(cell_type_pq_scores)[tissue_ids]]
                )
            )
            tissue_metrics[f"{tissue}"]["bPQ"] = float(
                np.nanmean(pq_scores[tissue_ids])
            )

        # calculate nuclei metrics
        nuclei_types = dataset_config["nuclei_types"]
        nuclei_metrics_d = {}
        nuclei_metrics_pq = {}
        nuclei_metrics_dq = {}
        nuclei_metrics_sq = {}
        for nuc_name, nuc_type in nuclei_types.items():
            if nuc_name.lower() == "background":
                continue
            nuclei_metrics_pq[nuc_name] = np.nanmean(
                [pq[nuc_type] for pq in cell_type_pq_scores]
            )
            nuclei_metrics_dq[nuc_name] = np.nanmean(
                [dq[nuc_type] for dq in cell_type_dq_scores]
            )
            nuclei_metrics_sq[nuc_name] = np.nanmean(
                [sq[nuc_type] for sq in cell_type_sq_scores]
            )
            f1_cell, prec_cell, rec_cell = cell_type_detection_scores(
                paired_true_type,
                paired_pred_type,
                unpaired_true_type,
                unpaired_pred_type,
                nuc_type,
            )
            nuclei_metrics_d[nuc_name] = {
                "f1_cell": f1_cell,
                "prec_cell": prec_cell,
                "rec_cell": rec_cell,
            }

        # logger.info final results
        # binary
        logger.info(f"{20*'*'} Binary Dataset metrics {20*'*'}")
        [logger.info(f"{f'{k}:': <25} {v}") for k, v in dataset_metrics.items()]
        # tissue -> the PQ values are bPQ values -> what about mBQ?
        logger.info(f"{20*'*'} Tissue metrics {20*'*'}")
        flattened_tissue = []
        for key in tissue_metrics:
            flattened_tissue.append(
                [
                    key,
                    tissue_metrics[key]["Dice"],
                    tissue_metrics[key]["Jaccard"],
                    tissue_metrics[key]["mPQ"],
                    tissue_metrics[key]["bPQ"],
                ]
            )
        logger.info(
            tabulate(
                flattened_tissue, headers=["Tissue", "Dice", "Jaccard", "mPQ", "bPQ"]
            )
        )
        # nuclei types
        logger.info(f"{20*'*'} Nuclei Type Metrics {20*'*'}")
        flattened_nuclei_type = []
        for key in nuclei_metrics_pq:
            flattened_nuclei_type.append(
                [
                    key,
                    nuclei_metrics_dq[key],
                    nuclei_metrics_sq[key],
                    nuclei_metrics_pq[key],
                ]
            )
        logger.info(
            tabulate(flattened_nuclei_type, headers=["Nuclei Type", "DQ", "SQ", "PQ"])
        )
        # nuclei detection metrics
        logger.info(f"{20*'*'} Nuclei Detection Metrics {20*'*'}")
        flattened_detection = []
        for key in nuclei_metrics_d:
            flattened_detection.append(
                [
                    key,
                    nuclei_metrics_d[key]["prec_cell"],
                    nuclei_metrics_d[key]["rec_cell"],
                    nuclei_metrics_d[key]["f1_cell"],
                ]
            )
        logger.info(
            tabulate(
                flattened_detection,
                headers=["Nuclei Type", "Precision", "Recall", "F1"],
            )
        )

        # save all folds
        all_metrics = {
            "dataset": dataset_metrics,
            "tissue_metrics": tissue_metrics,
            "nuclei_metrics_pq": nuclei_metrics_pq,
            "nuclei_metrics_d": nuclei_metrics_d,
        }

        # saving        
        with open(run_dir + "/inference_results.json", "w") as outfile:
            json.dump(all_metrics, outfile, indent=2)
