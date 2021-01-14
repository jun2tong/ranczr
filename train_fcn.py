import numpy as np
import time
import torch
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
        # img = torch.cat([x_mb, x_mb+mask_mb, mask_mb],dim=1)
        # img = torch.cat([x_mb, x_mb, x_mb],dim=1)
        img = x_mb.to(device)
        # x_mb = x_mb.to(device)
        # mask_mb = mask_mb.to(device)

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
                f"Grad: {grad_norm:.4f}"
            )
            print(print_str)
    scheduler.step()
    return losses.avg


def train_fn_seg(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    #     scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (x_mb, mask_mb, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img = torch.cat([x_mb, mask_mb[0], mask_mb[1]], dim=1)
        img = img.to(device)

        labels = labels.to(device)
        batch_size = labels.size(0)

        model_out = model(img)
        loss = criterion["cls"](model_out["cls"], labels)
        # seg_loss = criterion["seg"](model_out["seg_out"], mask_mb)
        # loss = cls_loss
        # record loss
        losses.update(loss.item(), batch_size)
        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 100 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "  # f"Breakdown: {seg_loss.detach().item():.4f}({cls_loss.detach().item():.4f}) "
                f"Grad: {grad_norm:.4f}"
            )
            print(print_str)
    scheduler.step()
    return losses.avg


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
        # img = torch.cat([x_mb, x_mb+mask_mb, mask_mb], dim=1)
        # img = torch.cat([x_mb, x_mb, x_mb], dim=1)
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
