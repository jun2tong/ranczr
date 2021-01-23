import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

import albumentations as a_transform
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from train_fcn import valid_fn_seg, train_fn_seg
from utils import get_score, init_logger
from optimizer import Ranger
from dataset import AnnotDataset, ValidDataset

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
    patience = 100
    num_workers = 4
    model_name = "effnetb2"
    size = 512
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

valid_transform = a_transform.Compose([a_transform.Resize(CFG.size, CFG.size), normalize, ToTensorV2()], p=1.0)


# Select fold - loop curfold
for curfold in CFG.trn_fold:
    LOGGER = init_logger(f"{CFG.model_name}-unet-fold{curfold}.log")
    curfold = 0
    trn_idx = folds[folds["fold"] != curfold].index
    val_idx = folds[folds["fold"] == curfold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    valid_labels = valid_folds[CFG.target_cols].values

    # Initialize train and valid dataset
    train_dataset = AnnotDataset(
        WORKDIR, train_folds, train_annot, flip_transform=train_transform, target_cols=CFG.target_cols
    )
    valid_dataset = ValidDataset(WORKDIR, valid_folds, transform=valid_transform, target_cols=CFG.target_cols)

    # Initialize dataloader for fast parsing
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

    # net = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    aux_params = dict(
        pooling="avg",  # one of 'avg', 'max'
        dropout=None,  # dropout ratio, default is None
        activation=None,  # activation function, default is None
        classes=CFG.target_size,  # define number of output labels
    )
    model = smp.Unet("efficientnet-b2", classes=1, aux_params=aux_params)
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = Ranger(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    CFG.T_max = int(CFG.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CFG.T_max * 0.7), gamma=0.9)

    criterion = nn.BCEWithLogitsLoss()

    best_loss = np.inf
    update_count = 0
    LOGGER.info(f"===== Begin Training fold{curfold} =====")
    for epc in range(CFG.epochs):
        start_time = time.time()
        if (epc + 1) == int(CFG.epochs * 0.7):
            print("Going into cosine regime.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(CFG.T_max * 0.3))
        avg_loss, seg_loss, cls_loss = train_fn_seg(train_loader, model, criterion, optimizer, epc, scheduler, device)

        avg_val_loss, preds = valid_fn_seg(valid_loader, model, criterion, device)
        score, scores = get_score(valid_labels, preds)
        elapsed = time.time() - start_time
        LOGGER.info(
            f"Epoch {epc+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  avg_seg_loss: {seg_loss:.4f} avg_cls_loss: {cls_loss:.4f} time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epc+1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}")
        if avg_val_loss < best_loss:
            update_count = 0
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
            LOGGER.info(f"Epoch {epc+1} - Save Best Loss: {best_loss:.4f} Model")
            torch.save({"model": model.state_dict(), "preds": preds}, f"{CFG.model_name}unet_best.pth")
        else:
            update_count += 1
            if update_count > CFG.patience:
                break
