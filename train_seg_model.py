import os
import sys
import ast
from albumentations.augmentations.transforms import CLAHE
import numpy as np
import pandas as pd
import cv2
import glob

# from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as a_transform
from albumentations.pytorch import ToTensorV2

# import matplotlib.pyplot as plt
from utils import AverageMeter
from dataset import AnnotDataset
from unet import UNet


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

WORKDIR = "../data/ranczr"

train_csv = pd.read_csv(os.path.join(WORKDIR, "train.csv"))
train_annot = pd.read_csv(os.path.join(WORKDIR, "train_annotations.csv"))
weird_uid = "1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280"
train_csv.loc[train_csv.StudyInstanceUID == weird_uid, "ETT - Abnormal"] = 0
train_csv.loc[train_csv.StudyInstanceUID == weird_uid, "CVC - Abnormal"] = 1
train_annot.loc[4344, "label"] = "CVC - Abnormal"


class CFG:
    debug = False
    print_freq = 100
    patience = 5
    num_workers = 4
    model_name = "resnext50_32x4d"
    size = 256
    scheduler = "CosineAnnealingLR"
    epochs = 30
    T_max = 30
    lr = 0.001
    min_lr = 0.000001
    batch_size = 16
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 1234
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
    trn_fold = [0, 1, 2, 3]
    train = True


folds = train_csv.copy()
Fold = GroupKFold(n_splits=CFG.n_fold)
groups = folds["PatientID"].values
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_cols], groups)):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)


normalize = a_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0, max_pixel_value=255.0)

train_transform = a_transform.Compose(
    [
        a_transform.RandomResizedCrop(CFG.size, CFG.size, scale=(0.9, 1.0), p=1.0),
        a_transform.HorizontalFlip(p=0.5),
        a_transform.CLAHE(clip_limit=(1, 4), p=0.5),
        a_transform.GaussianBlur(),
        normalize,
        ToTensorV2(),
    ],
    p=1.0,
)

valid_transform = a_transform.Compose([a_transform.Resize(256, 256), normalize, ToTensorV2()], p=1.0)


# Select fold - loop curfold
curfold = 0
trn_idx = folds[folds["fold"] != curfold].index
val_idx = folds[folds["fold"] == curfold].index

train_folds = folds.loc[trn_idx].reset_index(drop=True)
valid_folds = folds.loc[val_idx].reset_index(drop=True)

train_folds = train_folds[train_folds["StudyInstanceUID"].isin(train_annot["StudyInstanceUID"].unique())].reset_index(
    drop=True
)
valid_folds = valid_folds[valid_folds["StudyInstanceUID"].isin(train_annot["StudyInstanceUID"].unique())].reset_index(
    drop=True
)

valid_labels = valid_folds[CFG.target_cols].values

# train_uid = [x for x in train_folds["StudyInstanceUID"].values]
# valid_uid = [x for x in valid_folds["StudyInstanceUID"].values]
# Initialize train and valid dataset
train_dataset = AnnotDataset(
    WORKDIR, train_folds, train_annot, flip_transform=train_transform, target_cols=CFG.target_cols
)
valid_dataset = AnnotDataset(
    WORKDIR, valid_folds, train_annot, flip_transform=valid_transform, target_cols=CFG.target_cols
)

# Initialize dataloader for fast parsing
train_loader = DataLoader(
    train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    drop_last=False,
)

net = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)

criterion_recon = nn.BCEWithLogitsLoss()

best_loss = np.inf
update_count = 0
for epc in range(CFG.epochs):
    losses = AverageMeter()
    # scores = AverageMeter()
    for step, (img_mb, mask_mb, _) in enumerate(train_loader):
        img_mb = img_mb.to(device)

        # Model prediction
        mask_pred = net(img_mb)
        recon_loss = criterion_recon(mask_pred, mask_mb[0].to(device))
        batch_size = img_mb.size(0)
        # Record Loss
        losses.update(recon_loss.item(), batch_size)
        optimizer.zero_grad()
        recon_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1000)
        optimizer.step()

        if step % 100 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epc+1}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Grad: {grad_norm:.4f}"
            )
            print(print_str)

    scheduler.step()
    avg_valid = AverageMeter()
    # TODO: fix the reporting metric
    with torch.no_grad():
        for batch in valid_loader:
            img_mb, mask_mb = batch[0].to(device), batch[1][0].to(device)
            prediction = net(img_mb)
            mse = criterion_recon(prediction, mask_mb)
            avg_valid.update(mse.item(), img_mb.size(0))
        print(f"===> Epoch [{epc+1}] {avg_valid.val:.4f}({avg_valid.avg:.4f})")
        if avg_valid.avg < best_loss:
            update_count = 0
            best_loss = avg_valid.avg
            print(f"Epoch {epc+1} - Save Best Loss: {best_loss:.4f} Model")
            torch.save({"model": net.state_dict()}, f"seg-fcn8_best.pth")
        else:
            update_count += 1
            if update_count > CFG.patience:
                break
