import pdb
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from optimizer import Ranger
from dataset import TrainDataset, SegDataset, AnnotDataset, ValidDataset
from torch.utils.data import DataLoader
from train_fcn import train_fn, valid_fn, train_fn_seg, valid_fn_seg
from utils import get_score, init_logger
from losses import FocalLoss
from ranczr_models import CustomEffNet, CustomResNext, CustomXception, CustomInceptionV3
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import albumentations as a_transform
from albumentations.pytorch import ToTensorV2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
WORKDIR = "../data/ranczr"
# WORKDIR = "/home/jun/project/data/ranzcr-clip-catheter-line-classification"


def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    # train_folds = train_folds[train_folds['StudyInstanceUID'].isin(train_annot['StudyInstanceUID'].unique())].reset_index(drop=True)
    # valid_folds = valid_folds[valid_folds['StudyInstanceUID'].isin(train_annot['StudyInstanceUID'].unique())].reset_index(drop=True)

    valid_labels = valid_folds[CFG.target_cols].values
    # train_dataset = AnnotDataset(WORKDIR, train_folds, train_annot,
    #                            flip_transform=train_transform,
    #                            target_cols=CFG.target_cols)
    # valid_dataset = AnnotDataset(WORKDIR, valid_folds, train_annot,
    #                            flip_transform=valid_transform,
    #                            target_cols=CFG.target_cols)
    # train_dataset = SegDataset(WORKDIR, train_folds, flip_transform=train_transform, target_cols=CFG.target_cols)
    # valid_dataset = SegDataset(WORKDIR, valid_folds, flip_transform=valid_transform, target_cols=CFG.target_cols)
    train_dataset = TrainDataset(WORKDIR, train_folds, transform=train_transform, target_cols=CFG.target_cols)
    valid_dataset = ValidDataset(WORKDIR, valid_folds, transform=valid_transform, target_cols=CFG.target_cols)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    # model = CustomResNext(CFG.model_name, pretrained=True, target_size=CFG.target_size)
    if CFG.segment_model:
        raise NotImplementedError("Segmentation model not implemented")
        # model = SegModel(CFG.model_name, target_size=CFG.target_size)
    elif "efficient" in CFG.model_name:
        model = CustomEffNet(CFG.model_name, target_size=CFG.target_size)
    elif "xception" in CFG.model_name:
        model = CustomXception(None, target_size=CFG.target_size)
    elif "inception" in CFG.model_name:
        model = CustomInceptionV3(target_size=CFG.target_size)
    else:
        model = CustomResNext(CFG.model_name, target_size=CFG.target_size)

    model = nn.DataParallel(model)
    model.to(device)
    if CFG.resume:
        check_point = torch.load(CFG.resume_path)
        model.load_state_dict(check_point["model"])

    # optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    optimizer = Ranger(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    CFG.T_max = int(CFG.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CFG.T_max * 0.7), gamma=0.9)
    # scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs//2)

    # ====================================================
    # scheduler
    # ====================================================
    # def get_scheduler(optimizer):
    #     if CFG.scheduler=='ReduceLROnPlateau':
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
    #     elif CFG.scheduler=='CosineAnnealingLR':
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
    #     elif CFG.scheduler=='CosineAnnealingWarmRestarts':
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
    #     return scheduler

    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = {"cls": FocalLoss(alpha=1.5, logits=True), "seg": nn.BCEWithLogitsLoss()}
    criterion = {"cls": FocalLoss(alpha=1.8, gamma=1.2, logits=True), "seg": nn.BCEWithLogitsLoss()}

    best_score = 0.0
    best_loss = np.inf
    update_count = 0
    to_save = False
    for epoch in range(CFG.epochs):

        start_time = time.time()
        if (epoch + 1) == int(CFG.epochs * 0.7):
            print("Going into cosine regime.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(CFG.T_max * 0.3))
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        # avg_loss = train_fn_seg(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        # avg_val_loss, preds = valid_fn_seg(valid_loader, model, criterion, device)

        # scoring
        score, scores = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}")

        if avg_val_loss < best_loss or score > best_score:
            update_count = 0
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
            if score > best_score:
                best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model")
            torch.save({"model": model.state_dict(), "preds": preds}, f"{CFG.model_name}_fold{fold}_best.pth")
        else:
            update_count += 1
            if update_count >= CFG.patience:
                LOGGER.info(f"Early Stopped at Epoch {epoch+1}")
                break

    check_point = torch.load(f"{CFG.model_name}_fold{fold}_best.pth")
    for c in [f"pred_{c}" for c in CFG.target_cols]:
        valid_folds[c] = np.nan
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = check_point["preds"]

    return valid_folds


def main(folds):

    """
    Prepare: 1.train  2.folds
    """

    def get_result(result_df):
        preds = result_df[[f"pred_{c}" for c in CFG.target_cols]].values
        labels = result_df[CFG.target_cols].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f"Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}")

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(folds, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv("./oof_df.csv", index=False)


if __name__ == "__main__":

    class CFG:
        debug = False
        print_freq = 100
        num_workers = 4
        patience = 7
        segment_model = False
        model_name = "BiT-M-R50x3"
        backbone_name = "efficientnet-b5"
        resume = False
        resume_path = ""
        size = 512
        scheduler = "CosineAnnealingLR"
        epochs = 30
        T_max = 30
        lr = 0.0005
        min_lr = 0.000001
        batch_size = 8
        weight_decay = 1e-5
        gradient_accumulation_steps = 1
        max_grad_norm = 1000
        seed = 5468
        target_size = 11
        target_cols = [
            "ETT - Abnormal",
            "ETT - Borderline",
            "ETT - Normal",
            "NGT - Abnormal",
            "NGT - Borderline",
            "NGT - Incompletely Imaged",
            "NGT - Normal",
            "CVC - Abnormal",
            "CVC - Borderline",
            "CVC - Normal",
            "Swan Ganz Catheter Present",
        ]
        n_fold = 4
        trn_fold = [0]
        train = True

    normalize = a_transform.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0, max_pixel_value=255.0,
    )
    # normalize = a_transform.Normalize(mean=[0.485], std=[0.229], p=1.0, max_pixel_value=255.0)

    train_transform = a_transform.Compose(
        [
            a_transform.RandomResizedCrop(CFG.size, CFG.size, scale=(0.9, 1.0), p=1),
            a_transform.HorizontalFlip(p=0.5),
            a_transform.OneOf([a_transform.GaussNoise(var_limit=[10, 50]), a_transform.GaussianBlur()], p=0.5),
            a_transform.CLAHE(clip_limit=(1, 10), p=0.5),
            # a_transform.Rotate(limit=30),
            #    a_transform.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            a_transform.HueSaturationValue(p=0.5, hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
            a_transform.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=30),
            #    a_transform.CoarseDropout(p=0.2),
            #    a_transform.Cutout(p=0.2, max_h_size=8, max_w_size=8, fill_value=(0., 0., 0.), num_holes=8),
            #    a_transform.RandomSnow(p=0.3),
            #    a_transform.RandomContrast(),
            #    a_transform.RGBShift(),
            a_transform.OneOf(
                [a_transform.JpegCompression(), a_transform.Downscale(scale_min=0.1, scale_max=0.15),], p=0.2,
            ),
            normalize,
            ToTensorV2(),
        ],
        p=1.0,
    )

    valid_transform = a_transform.Compose([a_transform.Resize(CFG.size, CFG.size), normalize, ToTensorV2()], p=1.0)

    if not os.path.exists("./"):
        os.makedirs("./")
    LOGGER = init_logger(f"{CFG.model_name}-fold{CFG.trn_fold[0]}.log")

    train_csv = pd.read_csv(os.path.join(WORKDIR, "train.csv"))
    weird_uid = "1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280"
    train_csv.loc[train_csv.StudyInstanceUID == weird_uid, "ETT - Abnormal"] = 0
    train_csv.loc[train_csv.StudyInstanceUID == weird_uid, "CVC - Abnormal"] = 1
    train_annot = pd.read_csv(os.path.join(WORKDIR, "train_annotations.csv"))
    train_annot.loc[4344, "label"] = "CVC - Abnormal"

    folds = train_csv.copy()
    Fold = GroupKFold(n_splits=CFG.n_fold)
    groups = folds["PatientID"].values
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_cols], groups)):
        folds.loc[val_index, "fold"] = int(n)
    folds["fold"] = folds["fold"].astype(int)
    main(folds)
