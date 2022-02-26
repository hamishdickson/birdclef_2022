import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.cuda.amp import GradScaler
import transformers

from ast import literal_eval

from . import engine
from . import datasets
from . import models
from . import utils
import random
import warnings
from pprint import pprint


warnings.filterwarnings("ignore")

CONFIG = {
    "seed": 42,
    "n_fold": 5,
    "epochs": 10,
    "batch_size": 32,
    "n_accumulate": 1,
    "n_workers": 4,
    "model_save_name": "baseline",
    "model_name": "resnet34", # <---- just for you Eduardo
    "lr": 2e-4,
    "backbone_lr": 1e-4,
    "weight_decay": 0,
    "warmup": 0,
    "sample": False,
    "max_grad": 0,
    "num_class": 397,
    "train_period": 30.0,
    "infer_period": 30.0,
    "p": 0.5,
    "n_mels": 128,
    "mixup_p": 0.5,
    "mixup_alpha": 0.8
}


def train_loop(folds, fold=0):
    utils.set_seeds(CONFIG["seed"])

    print(f"training fold {fold}")
    writer = SummaryWriter()

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)

    print(len(train_folds), len(valid_folds))

    train_dataset = datasets.BirdClef2022Dataset(train_folds)
    valid_dataset = datasets.BirdClef2022Dataset(valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["n_workers"],
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        num_workers=CONFIG["n_workers"],
        pin_memory=True,
        drop_last=False
    )

    model = models.BaselineModel(CONFIG)
    model.cuda()

    optimizer_params = [
        {"params": model.att_model.head.parameters(), "lr": CONFIG['lr']},
        {"params": model.att_model.backbone.parameters(), "lr": CONFIG['backbone_lr']},
    ]

    optimizer = transformers.AdamW(
        optimizer_params,
        lr=CONFIG["lr"],
        eps=1e-8,
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=CONFIG['warmup'] * len(train_loader) * CONFIG['epochs'] / CONFIG["n_accumulate"],
        num_training_steps=len(train_loader) * CONFIG['epochs'] / CONFIG["n_accumulate"],
    )

    # scheduler = None

    scaler = GradScaler()

    mixup_p=0.0,
    mixup_alpha=0.5
    mixupper = utils.Mixup(p=CONFIG['mixup_p'], alpha=CONFIG['mixup_alpha'])
    thresholder = utils.ThresholdOptimizer(utils.row_wise_f1_score_micro)

    for epoch in range(CONFIG["epochs"]):

        train_f1 = engine.train_fn(
            epoch, train_loader, model, optimizer, scheduler, CONFIG, scaler, mixupper, thresholder
        )

        print(train_f1)

        valid_f1 = engine.valid_fn(model, valid_loader, thresholder)


        writer.add_scalar(
            "valid/f1",
            valid_f1,
            CONFIG["batch_size"] * len(train_loader) * (epoch + 1) / 32,
        )

        writer.add_scalar("train/f1", train_f1, epoch)

        print(f"results for epoch {epoch + 1}: f1 {valid_f1}")

    # torch.save(
    #     {"model": model.state_dict()},
    #     f"models/{CONFIG['model_name']}_{fold}.pth",
    # )

    # should probably build the oof file
    # oof.to_csv(f"models/oof_{CONFIG['model_name']}_{fold}.csv", index=False)

    del model
    return valid_f1


if __name__ == "__main__":
    print("training birds 2022")

    train_data = pd.read_csv("input/train_metadata_new.csv")

    CV = []

    for fold in range(CONFIG["n_fold"]):
        pprint(CONFIG)
        CV.append(train_loop(train_data, fold))
    print(f"final CVs {CV} = {np.mean(CV)}")
