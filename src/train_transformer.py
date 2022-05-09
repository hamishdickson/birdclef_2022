import os
import random
import pickle
import ast
import soundfile as sf
import pandas as pd
import librosa
import numpy as np
from sklearn import metrics

from pathlib import Path
import time

from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F

import transformers

from torch.cuda.amp import autocast, GradScaler

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils import general, metrics

from loss import loss_fn, mixup_criterion
from utils.general import cutmix, mixup
from configs import multicls_transformer


def fn(audio_path):
    waveform_orig, _ = sf.read("/mnt/datastore/birdclef_2022/input/train_audio/" + audio_path, always_2d=True)
    waveform_orig = waveform_orig[:, 0]

    spec = librosa.feature.melspectrogram(y=waveform_orig, sr=32000, n_mels=224, fmin=20, fmax=16000)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = spec[:,:3072 - 2]
    return spec


class BirdDataset(Dataset):
    def __init__(self, df, config, load_specs=None, mode='valid', do_secondary=False):
        self.df = df

        self.mode = mode

        self.config = config
        
        self.df["new_target"] = df["primary_label"] 
        
        if do_secondary:
            self.df["new_target"] = df["primary_label"] + " " + df["secondary_labels"].map(lambda x: " ".join(ast.literal_eval(x)))


        self.df["is_scored"] = self.df["new_target"].apply(
            lambda birds: any([bird in config.scored_birds for bird in birds.split()])
        )

        self.df["weight"] = self.df["is_scored"].apply(lambda x: config.scored_weight if x else 1)

        self.labels = self.df["new_target"].values
        self.max_length = config.max_length
        self.n_mels = config.n_mels

        self.target_columns = config.target_columns

        self.sr = 32000

        start_token = 0 
        end_token = 2
        self.pad_token = 1
        self.start_embed = np.full((self.n_mels, 1), start_token)
        self.end_embed = np.full((self.n_mels, 1), end_token)

        with open(load_specs, 'rb') as f:
            self.specs = pickle.load(f)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec = self.specs[idx]

        spec = spec[:,:self.max_length - 2] # start + end tokens

        pad_len = max(0, self.max_length - 2 - spec.shape[1])

        pad_left = random.randint(0, pad_len)
        pad_right = pad_len - pad_left

        if self.mode == 'train':
            padding_left = np.full((self.n_mels, pad_left), self.pad_token)
            padding_right = np.full((self.n_mels, pad_right), self.pad_token)

            embed = np.concatenate([padding_left, self.start_embed, spec, self.end_embed, padding_right], axis=1)
            attention_mask = [0] * pad_left + [1] * (self.max_length - pad_len) + [0] * pad_right
        else:
            padding = np.full((self.n_mels, pad_len), self.pad_token)
            embed = np.concatenate([self.start_embed, spec, self.end_embed, padding], axis=1)
            attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len

        targets = np.zeros(len(self.target_columns), dtype=float)
        for ebird_code in self.labels[idx].split():
            targets[self.target_columns.index(ebird_code)] = 1.0
        targets = torch.Tensor(targets)

        return {
            "embed": torch.Tensor(embed).transpose(0,1),
            "attention_mask": torch.Tensor(attention_mask),
            "target": targets
        }




class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()

        model_config = transformers.AutoConfig.from_pretrained(config.base_model_name)
        model_config.update(
                    {
                        "output_hidden_states": True,
                        "layer_norm_eps": 1e-7,
                        "add_pooling_layer": False,
                        "hidden_size": config.n_mels,
                        "num_attention_heads": config.num_attention_heads,
                        "max_position_embeddings": config.max_length,
                        "num_hidden_layers": config.hidden_layers,
                    }
                )
        self.base_model = transformers.AutoModel.from_config(model_config)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.n_mels, config.num_classes)


    def forward(self, spec_embeddings, pos_ids, mask):
        x = self.base_model(inputs_embeds=spec_embeddings, position_ids=pos_ids, attention_mask=mask)
        x = x.last_hidden_state
        x = self.dropout(x[:,0,:])
        return self.fc(x)



def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, config):
    print("start training")

    losses = metrics.AverageMeter()

    optimizer.zero_grad()

    model.train()

    tk0 = tqdm(train_loader, total=len(train_loader))
    for step, batch in enumerate(tk0):
        embed = batch['embed'].cuda()
        mask = batch['attention_mask'].cuda()
        targets = batch['target'].cuda()

        # todo check this thing
        input_shape = embed.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(
                    0, config.max_length , dtype=torch.long
                )
        pos_ids = position_ids.unsqueeze(0).expand(input_shape)
        pos_ids = pos_ids.cuda()

        is_mixup = False
        # if np.random.rand() < 0.5:
        #     embed, targets = mixup(embed, targets, config.mixup_alpha)
        #     is_mixup = True


        with autocast():
            logits = model(spec_embeddings=embed, pos_ids=pos_ids, mask=mask)
            # loss = loss_fn(logits, targets, alpha=0.25)
            if is_mixup:
                loss = mixup_criterion(logits, targets, alpha=0.25)
            else:
                loss = loss_fn(logits, targets, alpha=0.25)

        batch_size = embed.size(0)
        losses.update(loss.item(), batch_size)

        tk0.set_postfix(train_loss=losses.avg)

        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
                
    return losses.avg      



def valid_fn(model, valid_loader, config):
    model.eval()

    scores = metrics.MetricMeter()

    tk0 = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for batch in tk0:
            embed = batch['embed'].cuda()
            mask = batch['attention_mask'].cuda()
            targets = batch['target'].cuda()

            # todo check this thing
            input_shape = embed.size()[:-1]
            sequence_length = input_shape[1]
            position_ids = torch.arange(
                        0, config.max_length, dtype=torch.long
                    )
            pos_ids = position_ids.unsqueeze(0).expand(input_shape)
            pos_ids = pos_ids.cuda()

            with autocast():
                logits = model(spec_embeddings=embed, pos_ids=pos_ids, mask=mask)
                preds = logits.sigmoid()

            scores.update(targets, preds)

    return scores.avg


if __name__ == "__main__":
    general.set_seed()

    config = multicls_transformer.CFG

    df = pd.read_csv("data/train_metadata.csv")

    df_train = df[df['kfold'] != 0].reset_index(drop=True)
    df_valid = df[df['kfold'] == 0].reset_index(drop=True)

    build_data = True

    # if build_data:
    #     train_specs = Parallel(n_jobs=24)(delayed(fn)(audio_path) for audio_path in df_train.filename.values)
    #     valid_specs = Parallel(n_jobs=24)(delayed(fn)(audio_path) for audio_path in df_valid.filename.values)

    #     print(len(train_specs), len(valid_specs))
    #     with open('/mnt/datastore/birdclef_2022/train_mels_224v4.pkl', 'wb') as f:
    #         pickle.dump(train_specs, f)

    #     with open('/mnt/datastore/birdclef_2022/valid_mels_224v4.pkl', 'wb') as f:
    #         pickle.dump(valid_specs, f)

    #     del train_specs, valid_specs

    train_dataset = BirdDataset(
        df_train,
        config,
        mode='train',
        load_specs=f'/mnt/datastore/birdclef_2022/train_mels_224v4.pkl'
    )

    valid_dataset = BirdDataset(
        df_valid,
        config,
        load_specs=f'/mnt/datastore/birdclef_2022/valid_mels_224v4.pkl'
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_bs,
        shuffle=True,
        num_workers=config.nb_workers,
        pin_memory=True,
        drop_last=True,
    )     

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.valid_bs,
        shuffle=False,
        num_workers=config.nb_workers,
        pin_memory=True,
        drop_last=False,
    ) 


    model = TransformerModel(config)
    model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]
    optimizer_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

    optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr, weight_decay=config.weight_decay)

    scheduler = transformers.get_cosine_schedule_with_warmup( # note roberta paper uses linear
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * config.epochs,
        )

    scaler = GradScaler()

    writer = SummaryWriter()

    best_score = -np.inf
    save_path = None

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, scaler, config)

        valid_avg = valid_fn(model, valid_dataloader, config)

        for key in valid_avg.keys():
                if "best" in key:
                    print(
                        f"Epoch {epoch + 1} - valid_{key.replace('best', str(valid_avg[key][0]))}:{valid_avg[key][1]:0.5f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1} - valid_{key}:{valid_avg[key]:0.5f}"
                    )

        new_best = valid_avg.get("masked_f1_at_best", valid_avg["f1_at_best"])[1]

        writer.add_scalar("valid/best", new_best, epoch)

        _output_dir = (
            Path(config.output_dir)
            / f"{time.strftime('%D-%T').replace('/', '-')}-{config.exp_name}"
        )


        if new_best > best_score:
            _output_dir.mkdir(exist_ok=True, parents=True)
            new_save_path = (
                _output_dir
                / f"fold-{config.fold}-{config.base_model_name}-{config.exp_name}-epoch-{epoch}-f1-{new_best:.3f}-{valid_avg['f1_at_best'][1]:.3f}.bin"
            )
            print(f">>>>>>>> Model Improved From {best_score} ----> {new_best}")
            torch.save(model.state_dict(), new_save_path)
            if save_path:
                save_path.unlink()  # removes older checkpoints
            best_score = new_best
            save_path = new_save_path

