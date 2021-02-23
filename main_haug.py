import pdb
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from optimizer import Ranger
from dataset import TrainDataset, ValidDataset
from torch.utils.data import DataLoader
from train_fcn import train_fn, valid_fn, train_auc, valid_auc
from utils import get_score, init_logger
from losses import FocalLoss, DeepAUC
from ranczr_models import CustomAttention, EffNetWLF
from sklearn.model_selection import GroupKFold

import albumentations as a_transform
from albumentations.pytorch import ToTensorV2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
WORKDIR = "../data/ranczr"
# WORKDIR = "/home/jun/project/data/ranzcr-clip-catheter-line-classification"
torch.backends.cudnn.benchmark = True


def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    valid_labels = valid_folds[CFG.target_cols].values

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
    # model = CustomModel(CFG.model_name, target_size=CFG.target_size)
    if "efficient" in CFG.model_name:
        model = EffNetWLF(CFG.model_name, target_size=CFG.target_size, pretrained=not CFG.resume)
        # model = CustomAttention(CFG.model_name, target_size=CFG.target_size, pretrained=not CFG.resume)
    else:
        model = CustomAttention(CFG.model_name, target_size=CFG.target_size, pretrained=not CFG.resume)

    if CFG.resume:
        checkpoint = torch.load(CFG.resume_path)
        model.backbone.load_state_dict(checkpoint['model'])
        LOGGER.info(f"Loaded correct backbone for {CFG.model_name}")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Define expt_a, expt_b and alpha
    expt_a = torch.zeros(CFG.target_size, dtype=torch.float32, device=device, requires_grad=True)
    expt_b = torch.zeros(CFG.target_size, dtype=torch.float32, device=device, requires_grad=True)
    alpha = torch.zeros(CFG.target_size, dtype=torch.float32, device=device)+0.1
    # alpha.requires_grad = True

    pg_lr = [CFG.lr*0.5, CFG.lr, CFG.lr]
    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.Adam([{'params': model.module.backbone.parameters(), 'lr': pg_lr[0]},
                                      {'params': model.module.classifier.parameters(), 'lr': pg_lr[1]},
                                      {'params': model.module.local_fe.parameters(), 'lr': pg_lr[2]}], 
                                      weight_decay=CFG.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.backbone.parameters(), 'lr': pg_lr[0]},
                                      {'params': model.classifier.parameters(), 'lr': pg_lr[1]},
                                      {'params': model.local_fe.parameters(), 'lr': pg_lr[2]}], 
                                      weight_decay=CFG.weight_decay)
    aux_opt = torch.optim.Adam([expt_a, expt_b], lr=0.00002, weight_decay=1e-5, betas=(0.5, 0.999))
    # ====================================================
    # scheduler
    # ====================================================

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, min_lr=CFG.min_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=pg_lr, epochs=CFG.epochs, 
                                                    steps_per_epoch=len(train_loader)//CFG.gradient_accumulation_steps, 
                                                    final_div_factor = CFG.final_div_factor,
                                                    cycle_momentum=False)

    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = {"cls": FocalLoss(alpha=1.2, gamma=1.2, logits=True), "seg": nn.BCEWithLogitsLoss()}
    phat = torch.sum(train_dataset.labels, dim=0) / train_dataset.labels.shape[0]
    LOGGER.info(f"phat: {np.round(phat.numpy(), decimals=4)}")
    criterion = {"cls": nn.BCEWithLogitsLoss(), "auc": DeepAUC(phat.to(device))}

    best_score = 0.0
    best_loss = np.inf
    update_count = 0
    # grad_scaler = torch.cuda.amp.GradScaler()
    # grad_scaler = None
    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_auc(train_loader, model, expt_a, expt_b, alpha, 
                             criterion, optimizer, aux_opt, 
                             epoch, scheduler, device, CFG.gradient_accumulation_steps)
        # avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, CFG.gradient_accumulation_steps)
        # avg_loss = train_fn_seg(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        # avg_val_loss, preds = valid_fn(valid_loader, model, criterion["cls"], device)
        avg_val_loss, avg_cls_loss, avg_auc_loss, preds = valid_auc(valid_loader, model, expt_a, expt_b, alpha, criterion, device)

        # scoring
        score, scores = get_score(valid_labels, preds)

        # scheduler.step(score)
        elapsed = time.time() - start_time

        # LOGGER.info(f"Epoch {epoch+1} - scheduler lr: {scheduler.get_last_lr()}  time: {elapsed:.0f}s")
        # LOGGER.info(f"expt_a: {np.round(expt_a.data.cpu().numpy(), decimals=4)}")
        # LOGGER.info(f"expt_b: {np.round(expt_b.data.cpu().numpy(), decimals=4)}")
        # LOGGER.info(f"alpha: {np.round(alpha.data.cpu().numpy(), decimals=2)}")
        LOGGER.info(f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s")
        LOGGER.info(f"Breakdown: [{avg_cls_loss:.4f}][{avg_auc_loss:.4f}]")
        LOGGER.info(f"Epoch {epoch+1} - Score: {score:.4f}")
        # LOGGER.info(f"Epoch {epoch+1} - Scores: {np.round(scores, decimals=4)}")
        save_bs = False
        save_bl = False
        if score > best_score:
            best_score = score
            save_bs = True
        if avg_cls_loss < best_loss:
            best_loss = avg_cls_loss
            save_bl = True

        if save_bs or save_bl:
            LOGGER.info(f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Best Score: {best_score:.4f} Model")
            save_name = "val" if save_bl else "score"
            if torch.cuda.device_count() > 1:
                torch.save({"model": model.module.state_dict(), "preds": preds}, f"{CFG.model_name}_f{fold}_{save_name}.pth")
            else:
                torch.save({"model": model.state_dict(), "preds": preds}, f"{CFG.model_name}_f{fold}_{save_name}.pth")
        LOGGER.info(f"========== epoch {epoch+1} end ==========")

    check_point = torch.load(f"{CFG.model_name}_f{fold}_{save_name}.pth")
    for c in [f"pred_{c}" for c in CFG.target_cols]:
        valid_folds[c] = np.nan
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = check_point["preds"]

    return valid_folds


def main(folds, fold):

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
        _oof_df = train_loop(folds, fold)
        LOGGER.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df)

        # oof_df = pd.DataFrame()
        # for fold in range(CFG.n_fold):
        #     if fold in CFG.trn_fold:
        #         _oof_df = train_loop(folds, fold)
        #         oof_df = pd.concat([oof_df, _oof_df])
        #         LOGGER.info(f"========== fold: {fold} result ==========")
        #         get_result(_oof_df)
        # CV result
        # LOGGER.info(f"========== CV ==========")
        # get_result(oof_df)
        # save result
        # oof_df.to_csv("./oof_df.csv", index=False)


if __name__ == "__main__":

    class CFG:
        debug = False
        print_freq = 100
        num_workers = 4
        patience = 30
        model_name = "efficientnet_b5"
        backbone_name = "efficientnet-b2"
        resume = True
        resume_path = "pre-trained/efficientnet_b5.pth"
        size = 456
        # epochs = 25
        # lr = 0.00003
        epochs = 30
        lr = 0.00004
        min_lr = 0.000001
        final_div_factor = 50
        batch_size = 32
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
        n_fold = 5
        trn_fold = [3]
        train = True

    normalize = a_transform.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0, max_pixel_value=255.0,
    )
    # normalize = a_transform.Normalize(mean=[0.485], std=[0.229], p=1.0, max_pixel_value=255.0)
    augmentation = [
            a_transform.RandomResizedCrop(CFG.size, CFG.size, scale=(0.9, 1.0), p=1),
            # a_transform.SmallestMaxSize(CFG.size),
            # a_transform.RandomCrop(CFG.size, CFG.size),
            # a_transform.LongestMaxSize(max_size=CFG.size),
            # a_transform.PadIfNeeded(CFG.size, CFG.size),
            a_transform.HorizontalFlip(p=0.5),
            # a_transform.OneOf([a_transform.GaussNoise(var_limit=[10, 50]), a_transform.GaussianBlur()], p=0.5),
            a_transform.CLAHE(clip_limit=(1, 10), p=0.5),
            # a_transform.Rotate(limit=30),
            a_transform.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            # a_transform.HueSaturationValue(p=0.5, hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
            a_transform.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=30),
            # a_transform.CenterCrop(448, 448, p=1),
            # a_transform.CoarseDropout(p=0.2),
            # a_transform.Cutout(p=0.2, max_h_size=8, max_w_size=8, fill_value=(0., 0., 0.), num_holes=8),
            # a_transform.OneOf(
            #     [a_transform.JpegCompression(), a_transform.Downscale(scale_min=0.1, scale_max=0.15),], p=0.2,
            # ),
            normalize,
            ToTensorV2(),
        ]
    train_transform = a_transform.Compose(augmentation, p=1.0)

    valid_transform = a_transform.Compose([a_transform.Resize(CFG.size, CFG.size), normalize, ToTensorV2()], p=1.0)

    if not os.path.exists("./"):
        os.makedirs("./")

    train_csv = pd.read_csv(os.path.join(WORKDIR, "train.csv"))
    # weird_uid = "1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280"
    # train_csv.loc[train_csv.StudyInstanceUID == weird_uid, "ETT - Abnormal"] = 0
    # train_csv.loc[train_csv.StudyInstanceUID == weird_uid, "CVC - Abnormal"] = 1
    # train_annot = pd.read_csv(os.path.join(WORKDIR, "train_annotations.csv"))
    # train_annot.loc[4344, "label"] = "CVC - Abnormal"

    folds = train_csv.copy()
    Fold = GroupKFold(n_splits=CFG.n_fold)
    groups = folds["PatientID"].values
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_cols], groups)):
        folds.loc[val_index, "fold"] = int(n)
    folds["fold"] = folds["fold"].astype(int)
    # main(folds)
    for fold_idx in range(CFG.n_fold):
        if fold_idx in CFG.trn_fold:
            LOGGER = init_logger(f"{CFG.model_name}-f{fold_idx}.log")
            main(folds, fold_idx)
