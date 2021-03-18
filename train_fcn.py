import numpy as np
import time
import torch
import torch.nn.functional as F
from utils import AverageMeter, timeSince
import pdb


def train_auc(train_loader, model, expt_a, expt_b, alpha, criterion, optimizer, aux_opt, epoch, scheduler, device, gradient_acc_step=1):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    auc_losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    optimizer.zero_grad()
    aux_opt.zero_grad()
    for step, (x_mb, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = x_mb.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        img, targets_a, targets_b, lam = mixup_data(img, labels, 1.0, device)

        y_preds = model(img)
        cls_loss = mixup_criterion(criterion["cls"], y_preds, targets_a, targets_b, lam)
        # cls_loss = criterion['cls'](y_preds, labels)
        # auc_loss = criterion['auc'](y_preds, labels, expt_a, expt_b, alpha)
        auc_loss = criterion['auc'](y_preds, targets_a, expt_a, expt_b, alpha)*lam + criterion['auc'](y_preds, targets_b, expt_a, expt_b, alpha)*(1-lam)
        loss = auc_loss + cls_loss
        # loss = mixup_criterion(criterion["cls"], y_preds, targets_a, targets_b, lam)

        # record loss
        losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        auc_losses.update(auc_loss.item(), batch_size)

        if gradient_acc_step > 1:
            loss = loss / gradient_acc_step

        loss.backward()
        if (step+1) % gradient_acc_step == 0:
            optimizer.step()
            # aux_opt.step()
            # HACK to update alpha
            # if alpha.grad is not None:
            #     # alpha.data = alpha.data + 0.00002*alpha.grad.data
            #     alpha.data = torch.relu(alpha.data + 0.00005*alpha.grad.data)
            #     alpha.grad.data *= 0 
            optimizer.zero_grad()
            aux_opt.zero_grad()
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
        if step % 200 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Breakdown: [{cls_losses.avg:.4f}][{auc_losses.avg:.4f}]"
                # f"lr: {scheduler.get_last_lr()[0]:.7f}"
            )
            print(print_str)
    # scheduler.step()
    return losses.avg


def valid_auc(valid_loader, model, expt_a, expt_b, alpha, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    auc_losses = AverageMeter()
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
            y_preds = model(img)
            cls_loss = criterion['cls'](y_preds, labels)
            auc_loss = criterion['auc'](y_preds, labels, expt_a, expt_b, alpha)
            loss = cls_loss + auc_loss
        losses.update(loss.cpu().numpy(), batch_size)
        cls_losses.update(cls_loss.cpu().numpy(), batch_size)
        auc_losses.update(auc_loss.cpu().numpy(), batch_size)
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
    return losses.avg, cls_losses.avg, auc_losses.avg, predictions


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, gradient_acc_step=1):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    # gradient_acc_step = 2
    optimizer.zero_grad()
    for step, (x_mb, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = x_mb.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        img, targets_a, targets_b, lam = mixup_data(img, labels, 1.0, device)

        y_preds = model(img)
        # loss = criterion["cls"](y_preds, labels)
        loss = mixup_criterion(criterion["cls"], y_preds, targets_a, targets_b, lam)

        # record loss
        losses.update(loss.item(), batch_size)

        if gradient_acc_step > 1:
            loss = loss / gradient_acc_step

        loss.backward()
        if (step+1) % gradient_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
        if step % 200 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"lr: {scheduler.get_last_lr()[0]:.7f}"
            )
            print(print_str)
    # scheduler.step()
    return losses.avg


def train_ft(train_loader, model, criterion, optimizer, epoch, scheduler, device, gradient_acc_step=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    for step, (x_mb, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = x_mb.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        optimizer.zero_grad()
        y_preds = model(img)
        loss = criterion["cls"](y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)

        if gradient_acc_step > 1:
            loss = loss / gradient_acc_step

        loss.backward()
        if (step+1) % gradient_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % 200 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"lr: {scheduler.get_last_lr()[0]:.7f}"
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
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Breakdown: [{seg_losses.val:.4f}][{cls_losses.val:.4f}] "
                f"Grad: {grad_norm:.4f} lr: {scheduler.get_last_lr()[0]:.6f}"
            )
            print(print_str)
    return losses.avg, seg_losses.avg, cls_losses.avg


def train_fn_s2(train_loader, teacher, model, optimizer, epoch, scheduler, device, gradient_acc_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    feas_losses = AverageMeter()
    cls_losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    optimizer.zero_grad()
    for step, (img_mb, label_mb) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img_mb = img_mb.to(device)
        batch_size = img_mb.size(0)
        # features matching
        with torch.no_grad():
            teacher_feas = torch.sigmoid(teacher(img_mb))

        # Model predictions
        pred_y = model(img_mb)
        teach_loss = F.binary_cross_entropy_with_logits(pred_y, teacher_feas) 
        cls_loss = F.binary_cross_entropy_with_logits(pred_y, label_mb.to(device))
        loss = 0.7*teach_loss + 0.3*cls_loss

        # record loss
        losses.update(loss.item(), batch_size)
        feas_losses.update(teach_loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)

        if gradient_acc_step > 1:
            loss = loss / gradient_acc_step

        loss.backward()
        if (step+1) % gradient_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Record Loss
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 200 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Breakdown: [{feas_losses.val:.4f}][{cls_losses.val:.4f}] "
                f"lr: {scheduler.get_last_lr()[0]:.6f}"
            )
            print(print_str)
    # scheduler.step()
    return losses.avg, feas_losses.avg, cls_losses.avg


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
            y_preds = model(img)
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


def valid_fn_seg(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_losses = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (x_mb, mask_mb, label_mb, _) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # img = torch.cat([x_mb, mask_mb[0], mask_mb[1]], dim=1)
        img = x_mb.to(device)
        mask = mask_mb.to(device)
        labels = label_mb.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            pred_m, y_preds = model(img)
        loss = criterion(y_preds, labels)
        seg_loss = criterion(pred_m, mask)
        losses.update(loss.item(), batch_size)
        seg_losses.update(seg_loss.item(), batch_size)
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
                f"ClsLoss: {losses.val:.4f}({losses.avg:.4f}) "
                f"SegLoss: {seg_losses.val:.4f}({seg_losses.avg:.4f}) "
            )
            print(print_str)

    predictions = np.concatenate(preds)
    return losses.avg, predictions


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
