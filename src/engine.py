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

criterions = {
    "classification_clip": nn.BCEWithLogitsLoss(),
    "classification_frame": nn.BCEWithLogitsLoss(),
}


def loss_fn(outputs, batch):
    losses = {}
    losses["loss_clip"] = criterions["classification_clip"](
        torch.logit(outputs["output_clip"]), batch["loss_target"].cuda()
    )
    losses["loss_frame"] = criterions["classification_frame"](
        outputs["output_frame"].max(1)[0], batch["loss_target"].cuda()
    )
    losses["loss"] = losses["loss_clip"] + losses["loss_frame"] * 0.5
    return losses


def calc_metric(training_step_outputs, thresholder):
    y_true = []
    y_pred = []
    for tso in training_step_outputs:
        y_true.append(tso["target"])
        y_pred.append(tso["output_clip"])
    y_true = torch.cat(y_true).cpu().numpy().astype("int")
    y_pred = torch.cat(y_pred).cpu().detach().numpy()
    thresholder.fit(y_pred, y_true)
    coef = thresholder.coefficients()
    f1_score = thresholder.calc_score(y_pred, y_true, coef)
    f1_score_05 = thresholder.calc_score(y_pred, y_true, [0.5])
    f1_score_03 = thresholder.calc_score(y_pred, y_true, [0.3])
    return dict(
        train_coef=coef,
        train_f1_score=f1_score,
        train_f1_score_05=f1_score_05,
        train_f1_score_03=f1_score_03,
    )


def train_fn(
    epoch,
    train_loader,
    model,
    optimizer,
    scheduler,
    config,
    scaler,
    mixupper,
    thresholder,
):
    losses = utils.AverageMeter()
    losses_frame = utils.AverageMeter()
    losses_clip = utils.AverageMeter()

    optimizer.zero_grad()

    model.train()

    all_step_outputs = []

    tk0 = tqdm(train_loader, total=len(train_loader))

    for step, batch in enumerate(tk0):
        wave = batch["wave"].to("cuda")

        batch_size = wave.size(0)

        image = model.att_model.logmelspec_extractor(wave)[:, None]

        mixupper.init_lambda()

        step_output = {}

        image = mixupper.lam * image + (1 - mixupper.lam) * image.flip(0)

        with autocast():
            outputs = model(image)
            batch["loss_target"] = mixupper.lam * batch["loss_target"] + (
                1 - mixupper.lam
            ) * batch["loss_target"].flip(0)
            batch["target"] = mixupper.lam * batch["target"] + (
                1 - mixupper.lam
            ) * batch["target"].flip(0)

            train_loss = loss_fn(outputs, batch)

        step_output.update(train_loss)
        step_output.update({"output_clip": outputs["output_clip"]})
        step_output["target"] = batch["target"]

        loss = train_loss["loss"] / config["n_accumulate"]

        losses.update(train_loss["loss"].item(), batch_size)
        losses_frame.update(train_loss["loss_frame"].item(), batch_size)
        losses_clip.update(train_loss["loss_clip"].item(), batch_size)

        scaler.scale(loss).backward()

        if (step + 1) % config["n_accumulate"] == 0:

            if config["max_grad"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad"])

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        tk0.set_postfix(
            Epoch=epoch,
            train_loss=losses.avg,
            train_loss_frame=losses_frame.avg,
            train_loss_clip=losses_clip.avg,
        )

        all_step_outputs.append(step_output)

    metric = calc_metric(all_step_outputs, thresholder)

    gc.collect()

    return metric


def valid_fn(model, valid_loader, thresholder):
    model.eval()

    losses = utils.AverageMeter()
    losses_frame = utils.AverageMeter()
    losses_clip = utils.AverageMeter()

    tk0 = tqdm(valid_loader, total=len(valid_loader))

    steps = []

    for _, batch in enumerate(tk0):
        step_output = {}
        wave = batch["wave"].to("cuda")

        image = model.att_model.logmelspec_extractor(wave)[:, None]
        batch_size = wave.size(0)

        with torch.no_grad():
            with autocast():
                output = model(image)
                valid_loss = loss_fn(output, batch)

        losses.update(valid_loss["loss"].item(), batch_size)
        losses_frame.update(valid_loss["loss_frame"].item(), batch_size)
        losses_clip.update(valid_loss["loss_clip"].item(), batch_size)

        tk0.set_postfix(
            valid_loss=losses.avg,
            valid_loss_frame=losses_frame.avg,
            valid_loss_clip=losses_clip.avg,
        )

        step_output.update({"output_clip": output["output_clip"]})
        step_output["target"] = batch["target"]
        steps.append(step_output)

    score = calc_metric(steps, thresholder)

    return score
