import torch
import time
import numpy as np
import transformers

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from model import TimmSED
from dateset import WaveformDataset
from config import CFG, AudioParams
from utils.metrics import AverageMeter, MetricMeter
from loss import loss_fn


def train_fn(model, data_loader, device, optimizer, scheduler, do_mixup=False):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for data in tk0:
        optimizer.zero_grad()
        inputs = data['audio'].to(device)
        targets = data['targets'].to(device)
        with autocast(enabled=CFG.apex):
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
            inputs = data['audio'].to(device)
            targets = data['targets'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs["logit"], targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs["clipwise_output"])
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


class Trainer:
    def __init__(self, labels_df, fold=CFG.fold, device="cuda"):
        self.labels_df = labels_df
        self.fold = fold
        self.device = device

    def create_dataset(self, df, mode, batch_size, nb_workers, shuffle):
        dataset = WaveformDataset(df=df, labels_df=self.labels_df, mode=mode)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=nb_workers, pin_memory=True, shuffle=shuffle)
        return dataset, dataloader

    def train(self, train_df, val_df):
        print(f"Fold {self.fold} Training")

        train_dataset, train_dataloader = self.create_dataset(
            train_df, "train", CFG.train_bs, CFG.nb_workers, shuffle=True)
        valid_dateset, valid_dataloader = self.create_dataset(
            val_df, "val", CFG.valid_bs, CFG.nb_workers, shuffle=False)

        model = TimmSED(
            base_model_name=CFG.base_model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
            in_channels=CFG.in_channels)

        optimizer = transformers.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, eta_min=CFG.ETA_MIN, T_max=len(train_dataset) / CFG.train_bs * 18)

        model = model.to(self.device)

        best_score = -np.inf

        for epoch in range(CFG.epochs):
            print("Starting {} epoch...".format(epoch + 1))

            start_time = time.time()

            train_avg, train_loss = train_fn(
                model, train_dataloader, self.device, optimizer, scheduler,
                do_mixup=epoch < CFG.cutmix_and_mixup_epochs
            )

            valid_avg, valid_loss = valid_fn(model, valid_dataloader, self.device)

            elapsed = time.time() - start_time

            print(
                f'Epoch {epoch + 1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
            print(
                f"Epoch {epoch + 1} - train_f1_at_03:{train_avg['f1_at_03']:0.5f}  valid_f1_at_03:{valid_avg['f1_at_03']:0.5f}")
            print(
                f"Epoch {epoch + 1} - train_f1_at_05:{train_avg['f1_at_05']:0.5f}  valid_f1_at_05:{valid_avg['f1_at_05']:0.5f}")
            print(
                f"Epoch {epoch + 1} - train_f1_at_{train_avg['f1_at_best'][0]}:{train_avg['f1_at_best'][1]:0.5f}  valid_f1_at_{valid_avg['f1_at_best'][0]}:{valid_avg['f1_at_best'][1]:0.5f}")

            if valid_avg['f1_at_best'][1] > best_score:
                print(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['f1_at_best'][1]}")
                print(f"other scores here... {valid_avg['f1_at_03']}, {valid_avg['f1_at_05']}")
                torch.save(model.state_dict(), f'fold-{self.fold}.bin')
                best_score = valid_avg['f1_at_best'][1]


if __name__ == '__main__':
    import glob
    import ast
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from utils.general import set_seed

    set_seed(CFG.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_path = glob.glob(CFG.audios_path)

    df = pd.read_csv(CFG.train_metadata)

    df['new_target'] = df['primary_label'] + ' ' + df['secondary_labels'].map(
        lambda x: ' '.join(ast.literal_eval(x)))
    df['len_new_target'] = df['new_target'].map(lambda x: len(x.split()))

    path_df = pd.DataFrame(all_path, columns=['file_path'])
    path_df['filename'] = path_df['file_path'].map(
        lambda x: (x.split('/')[-2] + '/' + x.split('/')[-1]).replace(".npy", ""))

    df = pd.merge(df, path_df, on='filename')
    labels_df = pd.read_csv(CFG.train_labels)
    labels_df = \
        labels_df.merge(df[["new_target", "file_path"]], left_on="filepath", right_on="file_path", how="inner")[
            ["file_path", "new_target", "bird_pred", "seconds"]]
    labels_df = labels_df.set_index("file_path")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for n, (trn_index, val_index) in enumerate(kfold.split(df, df['primary_label'])):
        df.loc[val_index, 'kfold'] = int(n)
    df['kfold'] = df['kfold'].astype(int)

    train_df = df[df.kfold != CFG.fold].reset_index(drop=True)
    val_df = df[df.kfold == CFG.fold].reset_index(drop=True)
    Trainer(labels_df, device=device).train(train_df, val_df)
