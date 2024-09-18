import os
import cv2
import sys
import torch
import argparse
import numpy as np
from loguru import logger
import albumentations as A
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from models.cellmamba import CellMamba
from utils.get_loss import get_loss_fn
from utils.load_model import load_model
from datasets.pannuke import CustomCellSeg
from trainer.trainer_cellmamba import CellMambaTrainer
from utils.inference import run_patch_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Train CellMamba model for cell segmentation task.")
    parser.add_argument("--config", type=str, help="Configuration yaml file path.")
    parser.add_argument("--output", type=str, default="output", help="Configuration yaml file path.")
    return parser.parse_args()


if __name__ == "__main__":

    # Config logger
    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(sys.stderr, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add("info.log", level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)

    # Get config
    logger.info("Setup config file!")
    args = parse_args()
    config = OmegaConf.load(args.config)
    logger.info(args)

    # Setup Class
    tissue_types = np.load("tissue_types.npy", allow_pickle=True).item()

    # Setup transform
    logger.info("Setup data transformations!")
    trans = config.transformations
    input_shape = 256

    train_transform = A.Compose([
            A.RandomRotate90(p=trans.randomrotate90.p),
            A.HorizontalFlip(p=trans.horizontalflip.p),
            A.VerticalFlip(p=trans.verticalflip.p),
            A.Downscale(p=trans.downscale.p, scale_max=trans.downscale.scale, scale_min=trans.downscale.scale, interpolation=cv2.INTER_AREA),
            A.Blur(p=trans.blur.p, blur_limit=trans.blur.blur_limit),
            A.GaussNoise(p=trans.gaussnoise.p, var_limit=trans.gaussnoise.var_limit),
            A.ColorJitter(
                            p=trans.colorjitter.p,
                            brightness=trans.colorjitter.scale_setting,
                            contrast=trans.colorjitter.scale_setting,
                            saturation=trans.colorjitter.scale_color,
                            hue=trans.colorjitter.scale_color / 2,
                        ),
            A.Superpixels(
                            p=trans.superpixels.p,
                            p_replace=0.1,
                            n_segments=200,
                            max_size=int(input_shape / 2),
                        ),
            A.ZoomBlur(p=trans.zoomblur.p, max_factor=1.05),
            A.RandomSizedCrop(
                            min_max_height=(input_shape / 2, input_shape),
                            height=input_shape,
                            width=input_shape,
                            p=trans.randomsizedcrop.p,
                        ),
            A.ElasticTransform(p=trans.elastictransform.p, sigma=25, alpha=0.5, alpha_affine=None),
            A.Normalize(mean=trans.normalize.mean, std=trans.normalize.std)
      ])

    valid_transform = A.Compose([A.Normalize(mean=trans.normalize.mean, std=trans.normalize.std)])
    test_transform = A.Compose([A.Normalize(mean=trans.normalize.mean, std=trans.normalize.std)])

    logger.info(f"Train transform: \n{train_transform}")
    logger.info(f"Val transform: \n{valid_transform}")
    logger.info(f"Test transform: \n{test_transform}")

    # Setup dataset
    logger.info("Setup custom dataset!")
    train_data = CustomCellSeg(image_dir=config.data.train.image_dir,
                               label_dir=config.data.train.label_dir,
                               class_names=tissue_types,
                               transforms=train_transform)

    valid_data = CustomCellSeg(image_dir=config.data.val.image_dir,
                               label_dir=config.data.val.label_dir,
                               class_names=tissue_types,
                               transforms=valid_transform)

    test_data = CustomCellSeg(image_dir=config.data.test.image_dir,
                              label_dir=config.data.test.label_dir,
                              class_names=tissue_types,
                              transforms=test_transform)

    # Setup training
    logger.info("Setup training!")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(config, num_classes=len(tissue_types)).to(device)
    model.freeze_encoder()
    loss_fn_dict = get_loss_fn()
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.85, 0.95), weight_decay=0.0001, lr = 0.0003)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    logger.info(f"Model: \n{model}")
    
    # Setup data loader
    train_dataloader = DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=config.training.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=config.training.batch_size, shuffle=False)

    # Setup trainer
    trainer = CellMambaTrainer(model=model,
                               loss_fn_dict=loss_fn_dict,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               device=device,
                               num_classes=6, # This is nuclei class sr please fix the name for me
                               patience=config.patience,
                               logdir=args.output, # a
                               )

    # Fit
    logger.info("Trainer fit!")
    trainer.fit(epochs=config.training.epochs,
                train_dataloader=train_dataloader,
                val_dataloader=valid_dataloader,
                metric_init=None,
                eval_every=config.training.eval_every)
    
    try:
        logger.info("Best checkpoint loaded!")
        model.load_state_dict(torch.load("latest_checkpoint.pth", weights_only=True))
    except:
        logger.info("Load checkpoint failed!")

    # Infer
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
    run_patch_inference(model=model,
                        run_dir=".",
                        inference_dataloader=test_dataloader,
                        device="cuda",
                        logger=logger,
                        dataset_config=dataset_config,
                        generate_plots=False)
