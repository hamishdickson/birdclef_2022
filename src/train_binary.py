import torch

from configs.binary import CFG
from dataset import BinaryDataset
from engine import Trainer
from utils.binary_utils import create_binary_df
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


if __name__ == "__main__":
    set_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df = create_binary_df()

    train_dataset, train_dataloader = create_dataset(
        train_df, "train", CFG.train_bs, CFG.nb_workers, shuffle=True
    )
    valid_dateset, valid_dataloader = create_dataset(
        val_df, "val", CFG.valid_bs, CFG.nb_workers, shuffle=False
    )

    Trainer(CFG, device=device).train(train_dataloader, valid_dataloader)
