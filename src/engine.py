import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from . import model as models
from .utils.metrics import AverageMeter, MetricMeter


def train_fn(
    model, data_loader, device, optimizer, scheduler, do_mixup=False, use_apex=True, gd_steps=1
):
    model.train()
    scaler = GradScaler(enabled=use_apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for i, data in enumerate(tk0):
        optimizer.zero_grad()
        inputs = data["audio"].to(device, non_blocking=True)
        targets = data["targets"].to(device, non_blocking=True)
        weights = None
        if "weights" in data.keys():
            weights = data["weights"].to(device, non_blocking=True)

        with autocast(enabled=use_apex):
            outputs = model(inputs, targets=targets, do_mixup=do_mixup, weights=weights)
            loss = outputs["loss"]
        # gradient accumulation
        loss = loss / gd_steps
        scaler.scale(loss).backward()
        if (i + 1) % gd_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs["clipwise_output"], mask=data.get("is_scored"))
        tk0.set_postfix(loss=losses.avg, lr=optimizer.param_groups[0]["lr"])
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
            outputs = model(inputs, targets)
            loss_meter.update(outputs["loss"].item(), inputs.size(0))
            score_meter.update(targets, outputs["clipwise_output"], mask=data.get("is_scored"))
            tk0.set_postfix(loss=loss_meter.avg)
    return score_meter, loss_meter


class Trainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self._output_dir = Path(self.cfg.output_dir) / f"{self.cfg.exp_name}"

    def validate(self, valid_dataloader, model_path, loss_meter=None, score_meter=None):
        model = getattr(models, self.cfg.meta_model_name)(
            cfg=self.cfg,
            base_model_name=self.cfg.base_model_name,
            pretrained=self.cfg.pretrained,
            num_classes=self.cfg.num_classes,
        )
        model.load_state_dict(torch.load(model_path, "cpu"), strict=True)
        print(f"Loaded state dict from {model_path}")
        model.to(self.device)
        start_time = time.time()
        valid_score, valid_loss = valid_fn(
            model, valid_dataloader, self.device, loss_meter, score_meter
        )
        elapsed = time.time() - start_time
        print(f"Time: {elapsed:.0f}s - avg_val_loss: {valid_loss.avg:.5f}")
        return valid_score

    def train(self, train_dataloader, valid_dataloader):
        print(f"Fold {self.cfg.fold} Training")

        model = getattr(models, self.cfg.meta_model_name)(
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

        params = []
        no_wd = []
        lr = self.cfg.LR * self.cfg.train_bs * self.cfg.grad_acc_steps / 32
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_wd.append(n)
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            weight_decay = self.cfg.WEIGHT_DECAY
            if "bias" in key:
                weight_decay = 0.0
            if any([key.startswith(no_wd_key) for no_wd_key in no_wd]):
                weight_decay = 0.0
            if "gain" in key or "skipinit_gain" in key:
                weight_decay = 0.0
            if "pos_embed" in key or "cls_token" in key or "dist_token" in key:
                weight_decay = 0.0
            if "relative_position_bias_table" in key:
                weight_decay = 0.0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = torch.optim.AdamW(params, eps=1e-6)

        num_epochs = self.cfg.epochs
        num_warmup_epochs = self.cfg.warmup_epochs
        grad_acc_steps = self.cfg.grad_acc_steps
        num_training_batches = len(train_dataloader)
        num_training_steps = (num_epochs * num_training_batches) // grad_acc_steps
        num_warmup_steps = (num_warmup_epochs * num_training_batches) // grad_acc_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
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
                gd_steps=grad_acc_steps,
            )

            valid_avg, valid_loss = valid_fn(model, valid_dataloader, self.device)
            valid_loss = valid_loss.avg
            valid_avg = valid_avg.avg

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch + 1} - avg_train_loss: {train_loss:.5f}"
                f" avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s"
                f" lr: {optimizer.param_groups[0]['lr']}"
            )
            for key in valid_avg.keys():
                if "best" in key:
                    print(
                        f"Epoch {epoch + 1} - train_{key.replace('best', str(train_avg[key][0]))}:{train_avg[key][1]:0.5f}  valid_{key.replace('best', str(valid_avg[key][0]))}:{valid_avg[key][1]:0.5f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1} - train_{key}:{train_avg[key]:0.5f}"
                        " valid_{key}:{valid_avg[key]:0.5f}"
                    )

            new_best = valid_avg.get("masked_f1_at_best", valid_avg["f1_at_best"])[1]
            if new_best > best_score:
                self._output_dir.mkdir(exist_ok=True, parents=True)
                new_save_path = (
                    self._output_dir
                    / f"fold-{self.cfg.fold}-{self.cfg.base_model_name}-{self.cfg.exp_name}"
                    f"-epoch-{epoch}-f1-{new_best:.3f}-{valid_avg['f1_at_best'][1]:.3f}.bin"
                )
                print(f">>>>>>>> Model Improved From {best_score} ----> {new_best}")
                torch.save(model.state_dict(), new_save_path)
                if save_path:
                    save_path.unlink()  # removes older checkpoints
                best_score = new_best
                save_path = new_save_path
        return save_path, best_score
