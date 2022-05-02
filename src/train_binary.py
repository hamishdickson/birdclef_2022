from pathlib import Path

import pandas as pd
import torch

from configs.binary import CFG
from dataset import BinaryDataset
from engine import Trainer
from utils.general import set_seed


def create_dataset(df, mode, batch_size, nb_workers, shuffle):
    dataset = BinaryDataset(
        df=df,
        sr=CFG.sample_rate,
        duration=CFG.period,
        mode=mode,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nb_workers,
        pin_memory=True,
        shuffle=shuffle,
    )
    return dataset, dataloader


def create_df(csv_path, audio_root):
    """Creates a DF for external datasets"""
    audio_root = Path(audio_root)
    df = pd.read_csv(csv_path)
    df["filepath"] = [str(audio_root / f"{x}.wav") for x in df.itemid]
    df["label"] = df.hasbird
    df["filename"] = df["itemid"].apply(lambda x: str(x) + ".wav")
    return df


def create_df_2021(csv_path, audio_root):
    audio_root = Path(audio_root)
    df = pd.read_csv(csv_path)
    mapping = {"_".join(x.stem.split("_")[:-1]): x.stem for x in audio_root.glob("*.ogg")}
    df["filepath"] = [
        str(audio_root / f"{mapping['_'.join(x.split('_')[:-1])]}.ogg") for x in df["row_id"]
    ]
    df["label"] = df.birds.apply(lambda x: int(x != "nocall"))
    df["filename"] = df["row_id"].apply(lambda x: x + ".ogg")
    return df


if __name__ == "__main__":
    set_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = []
    for dataset_root, dataset_csv in CFG.train_data:
        df = create_df(dataset_csv, dataset_root)
        train_df.append(df[["filepath", "label"]])
        print(f"{len(df)} samples added to train!")
    train_df = pd.concat(train_df)
    assert train_df.filepath.duplicated().sum() == 0
    print(
        (
            f"Total train size {len(train_df)}. "
            f"Pos count {(train_df.label == 1).sum()}, neg count {(train_df.label == 0).sum()}"
        )
    )

    val_df = create_df_2021(CFG.val_data[1], CFG.val_data[0])
    print(
        (
            f"Total val size {len(val_df)}. "
            f"Pos count {(val_df.label == 1).sum()}, neg count {(val_df.label == 0).sum()}"
        )
    )

    train_dataset, train_dataloader = create_dataset(
        train_df, "train", CFG.train_bs, CFG.nb_workers, shuffle=True
    )
    valid_dateset, valid_dataloader = create_dataset(
        val_df, "val", CFG.valid_bs, CFG.nb_workers, shuffle=False
    )

    Trainer(CFG, device=device).train(train_dataloader, valid_dataloader)
