import time

import numpy as np
import torch
import transformers
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from loss import loss_fn
from model import TimmSED
from utils.metrics import AverageMeter, MetricMeter


def train_fn(model, data_loader, device, optimizer, scheduler, do_mixup=False, use_apex=True):
    model.train()
    scaler = GradScaler(enabled=use_apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for data in tk0:
        optimizer.zero_grad()
        inputs = data["audio"].to(device)
        targets = data["targets"].to(device)
        with autocast(enabled=use_apex):
            outputs = model(inputs, targets=targets, do_mixup=do_mixup)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs["clipwise_output"])
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tk0:
            inputs = data["audio"].to(device)
            targets = data["targets"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs["logit"], targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs["clipwise_output"])
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


class Trainer:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device

    def train(self, train_dataloader, valid_dataloader):
        print(f"Fold {self.cfg.fold} Training")

        model = TimmSED(
            cfg=self.cfg,
            base_model_name=self.cfg.base_model_name,
            pretrained=self.cfg.pretrained,
            num_classes=self.cfg.num_classes,
        )

        optimizer = transformers.AdamW(
            model.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            eta_min=self.cfg.ETA_MIN,
            T_max=len(train_dataloader.dataset) / self.cfg.train_bs * 18,
        )

        model = model.to(self.device)

        best_score = -np.inf

        for epoch in range(self.cfg.epochs):
            print("Starting {} epoch...".format(epoch + 1))

            start_time = time.time()

            train_avg, train_loss = train_fn(
                model,
                train_dataloader,
                self.device,
                optimizer,
                scheduler,
                do_mixup=epoch < self.cfg.cutmix_and_mixup_epochs,
                use_apex=self.cfg.apex,
            )

            valid_avg, valid_loss = valid_fn(model, valid_dataloader, self.device)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch + 1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s"
            )
            print(
                f"Epoch {epoch + 1} - train_f1_at_03:{train_avg['f1_at_03']:0.5f}  valid_f1_at_03:{valid_avg['f1_at_03']:0.5f}"
            )
            print(
                f"Epoch {epoch + 1} - train_f1_at_05:{train_avg['f1_at_05']:0.5f}  valid_f1_at_05:{valid_avg['f1_at_05']:0.5f}"
            )
            print(
                f"Epoch {epoch + 1} - train_f1_at_{train_avg['f1_at_best'][0]}:{train_avg['f1_at_best'][1]:0.5f}  valid_f1_at_{valid_avg['f1_at_best'][0]}:{valid_avg['f1_at_best'][1]:0.5f}"
            )

            if valid_avg["f1_at_best"][1] > best_score:
                print(
                    f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['f1_at_best'][1]}"
                )
                print(f"other scores here... {valid_avg['f1_at_03']}, {valid_avg['f1_at_05']}")
                torch.save(model.state_dict(), f"fold-{self.cfg.fold}.bin")
                best_score = valid_avg["f1_at_best"][1]
