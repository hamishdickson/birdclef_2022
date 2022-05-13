import imp
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from loss import loss_fn
from model import TimmSED
from utils.metrics import AverageMeter, MetricMeter

from torch.utils.tensorboard import SummaryWriter


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
        weights = None
        if "weights" in data.keys():
            weights = data["weights"].to(device)

        with autocast(enabled=use_apex):
            outputs = model(inputs, targets=targets, do_mixup=do_mixup, weights=weights)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs["clipwise_output"], mask=data.get("is_scored"))
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device, loss_meter=None, score_meter=None):
    if loss_meter is None:
        loss_meter = AverageMeter()
    if score_meter is None:
        score_meter = MetricMeter()

    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tk0:
            inputs = data["audio"].to(device)
            targets = data["targets"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs["logit"], targets)
            loss_meter.update(loss.item(), inputs.size(0))
            score_meter.update(targets, outputs["clipwise_output"], mask=data.get("is_scored"))
            tk0.set_postfix(loss=loss_meter.avg)
    return score_meter, loss_meter


class Trainer:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self._output_dir = (
            Path(self.cfg.output_dir)
            / f"{time.strftime('%D-%T').replace('/', '-')}-{self.cfg.exp_name}"
        )

    def train(self, train_dataloader, valid_dataloader):
        print(f"Fold {self.cfg.fold} Training")

        writer = SummaryWriter()

        model = TimmSED(
            cfg=self.cfg,
            base_model_name=self.cfg.base_model_name,
            pretrained=self.cfg.pretrained,
            num_classes=self.cfg.num_classes,
        )

        if self.cfg.starting_weights:
            weigths_dict = torch.load(self.cfg.starting_weights)

            if self.cfg.load_up_to_layer:
                for i, key in enumerate(weigths_dict.keys()):
                    if (
                        "encoder.blocks" in key
                        and int(key.split(".")[2]) > self.cfg.load_up_to_layer
                    ):
                        to_remove = i
                        break

                weigths_dict = {
                    k: v for i, (k, v) in enumerate(weigths_dict.items()) if i < to_remove
                }

            loaded_keys = model.load_state_dict(weigths_dict, strict=False)
            print(f"Loaded weights from {self.cfg.starting_weights}")
            print(loaded_keys)

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
        save_path = None

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
            valid_loss = valid_loss.avg
            valid_avg = valid_avg.avg

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch + 1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s"
            )
            for key in valid_avg.keys():
                if "best" in key:
                    print(
                        f"Epoch {epoch + 1} - train_{key.replace('best', str(train_avg[key][0]))}:{train_avg[key][1]:0.5f}  valid_{key.replace('best', str(valid_avg[key][0]))}:{valid_avg[key][1]:0.5f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1} - train_{key}:{train_avg[key]:0.5f}  valid_{key}:{valid_avg[key]:0.5f}"
                    )

            new_best = valid_avg.get("masked_f1_at_best", valid_avg["f1_at_best"])[1]
            writer.add_scalar("valid/best", new_best, epoch)
            if new_best > best_score:
                self._output_dir.mkdir(exist_ok=True, parents=True)
                new_save_path = (
                    self._output_dir
                    / f"fold-{self.cfg.fold}-{self.cfg.base_model_name}-{self.cfg.exp_name}-epoch-{epoch}-f1-{new_best:.3f}-{valid_avg['f1_at_best'][1]:.3f}.bin"
                )
                print(f">>>>>>>> Model Improved From {best_score} ----> {new_best}")
                torch.save(model.state_dict(), new_save_path)
                if save_path:
                    save_path.unlink()  # removes older checkpoints
                best_score = new_best
                save_path = new_save_path
        return save_path, best_score
