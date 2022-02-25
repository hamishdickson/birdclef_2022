import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from . import utils
from . import datasets

import pandas as pd

from sklearn.metrics import f1_score

import gc

from torch.cuda.amp import autocast

def calc_metric(preds, targets):
    # todo
    return 0

def train_fn(
    epoch, train_loader, model, optimizer, criterion, scheduler, config, scaler
):
    losses = utils.AverageMeter()
    f1s = utils.AverageMeter()

    optimizer.zero_grad()

    model.train()

    tk0 = tqdm(train_loader, total=len(train_loader))

    for step, batch in enumerate(tk0):
        images = batch["image"].to(config["device"], dtype=torch.long)
        targets = batch["targets"].to(config["device"], dtype=torch.long)

        batch_size = images.size(0)

        with autocast():
            output = model(images)

        loss = criterion(output, targets)

        loss = loss / config["n_accumulate"]

        metric = calc_metric(output, targets)

        losses.update(loss.item(), batch_size)
        f1s.update(metric, batch_size)

        scaler.scale(loss).backward()

        if (step + 1) % config["n_accumulate"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad"])

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        tk0.set_postfix(Epoch=epoch, train_loss=losses.avg, f1=f1s.avg)

    gc.collect()

    return f1s.avg


def valid_fn(model, valid_loader, config):
    model.eval()

    tk0 = tqdm(valid_loader, total=len(valid_loader))

    scores = (0, {}) # (comp_metric, f1s_for_each_of_21_birds)
    submission = None

    for step, batch in enumerate(tk0):
        images = batch["image"].to(config["device"], dtype=torch.long)
        targets = batch["targets"].to(config["device"], dtype=torch.long)

        with torch.no_grad():
            with autocast():
                output = model(images)

    # TODO .... the hard bit

    return scores, submission
