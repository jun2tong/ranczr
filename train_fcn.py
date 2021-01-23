import numpy as np
import time
import torch
import torch.nn.functional as F
import pdb
from utils import AverageMeter, timeSince


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    #     scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (x_mb, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = x_mb.to(device)

        labels = labels.to(device)
        batch_size = labels.size(0)

        y_preds = model(img)
        loss = criterion["cls"](y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
        if step % 100 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Grad: {grad_norm:.4f} lr: {scheduler.get_last_lr()[0]:.6f}"
            )
            print(print_str)
    # scheduler.step()
    return losses.avg


def train_fn_seg(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_losses = AverageMeter()
    cls_losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    for step, (img_mb, mask_mb, label_mb, hmask_mb) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img_mb = img_mb.to(device)
        batch_size = img_mb.size(0)
        # Model prediction
        mask_pred, pred_y = model(img_mb)
        seg_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_mb.to(device), reduction="none")
        cls_loss = F.binary_cross_entropy_with_logits(pred_y, label_mb.to(device))
        seg_loss = torch.mean(torch.mean(seg_loss.view(batch_size, -1), dim=1) * hmask_mb.to(device))
        loss = seg_loss + cls_loss

        # Record Loss
        losses.update(loss.item(), batch_size)
        seg_losses.update(seg_loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
        optimizer.step()
        scheduler.step()

        # record loss
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 200 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "  # f"Breakdown: {seg_loss.detach().item():.4f}({cls_loss.detach().item():.4f}) "
                f"Grad: {grad_norm:.4f} lr: {scheduler.get_last_lr()[0]:.6f}"
            )
            print(print_str)
    return losses.avg, seg_losses.avg, cls_losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (x_mb, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = x_mb.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            # model_out = model(img)
            # y_preds = model_out["cls"]
            y_preds = model(img)
        loss = criterion["cls"](y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to("cpu").numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % 100 == 0 or step == (len(valid_loader) - 1):
            print_str = (
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(valid_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
            )
            print(print_str)

    predictions = np.concatenate(preds)
    return losses.avg, predictions


def valid_fn_seg(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (x_mb, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # img = torch.cat([x_mb, mask_mb[0], mask_mb[1]], dim=1)
        img = x_mb.to(device)

        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            # model_out = model(img)
            # y_preds = model_out["cls"]
            _, y_preds = model(img)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to("cpu").numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % 200 == 0 or step == (len(valid_loader) - 1):
            print_str = (
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(valid_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
            )
            print(print_str)

    predictions = np.concatenate(preds)
    return losses.avg, predictions
