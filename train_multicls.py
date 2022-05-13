import ast
import glob
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from src.configs.multicls import CFG
from src.dataset import WaveformDataset
from src.engine import Trainer
from src.utils.general import set_seed


def create_dataset(df, labels_df, mode, batch_size, nb_workers, shuffle):
    dataset = WaveformDataset(
        df=df,
        labels_df=labels_df,
        sr=CFG.sample_rate,
        duration=CFG.period,
        target_columns=CFG.target_columns,
        mode=mode,
        split_audio_root=CFG.split_audios_path,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nb_workers,
        pin_memory=True,
        shuffle=shuffle,
    )
    return dataset, dataloader


def create_df():
    all_path = glob.glob(CFG.audios_path)

    df = pd.read_csv(CFG.train_metadata)
    if not ("new_target" in df.columns and "kfold" in df.columns):
        print("DataFrame not processed. Processing now...")
        df["new_target"] = (
            df["primary_label"]
            + " "
            + df["secondary_labels"].map(lambda x: " ".join(ast.literal_eval(x)))
        )
        df["len_new_target"] = df["new_target"].map(lambda x: len(x.split()))

        df["is_scored"] = df["new_target"].apply(
            lambda birds: any([bird in CFG.scored_birds for bird in birds.split()])
        )

        df["weight"] = df["is_scored"].apply(lambda x: CFG.scored_weight if x else 1)

        path_df = pd.DataFrame(all_path, columns=["file_path"])
        path_df["filename"] = path_df["file_path"].map(
            lambda x: (x.split("/")[-2] + "/" + x.split("/")[-1]).replace(".npy", "")
        )

        df = pd.merge(df, path_df, on="filename")

        kfold = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
        for n, (trn_index, val_index) in enumerate(kfold.split(df, df["primary_label"])):
            df.loc[val_index, "kfold"] = int(n)
        df["kfold"] = df["kfold"].astype(int)
        df.to_csv(str(Path(__file__).parent / "../data/train_metadata.csv"), index=False)

    # df = df[df["secondary_labels"].apply(lambda x: len(ast.literal_eval(x))) == 0]
    # df = df.head(1000)  # for deebug
    labels_df = pd.read_csv(CFG.train_labels)
    labels_df = labels_df.merge(
        df[["new_target", "file_path"]], left_on="filepath", right_on="file_path", how="inner"
    )[["file_path", "new_target", "bird_pred", "seconds"]]
    labels_df = labels_df.set_index("file_path")
    return df, labels_df


def train_fold():
    set_seed(
        CFG.seed + CFG.fold
    )  # make sure each fold has different seed set, dataset split seed set separately
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, labels_df = create_df()

    train_df = df[df.kfold != CFG.fold].reset_index(drop=True)
    val_df = df[df.kfold == CFG.fold].reset_index(drop=True)

    train_dataset, train_dataloader = create_dataset(
        train_df, labels_df, "train", CFG.train_bs, CFG.nb_workers, shuffle=True
    )
    valid_dateset, valid_dataloader = create_dataset(
        val_df, labels_df, "val", CFG.valid_bs, CFG.nb_workers, shuffle=False
    )

    return Trainer(CFG, device=device).train(train_dataloader, valid_dataloader)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from eval_multicls import eval_kfold

    parser = ArgumentParser()
    parser.add_argument("--multi_fold", action="store_true")
    parser.add_argument("--nb_folds", type=int, default=5)
    args = parser.parse_args()

    if args.multi_fold:
        folds = range(args.nb_folds)

        model_paths = {}
        for fold in folds:
            CFG.fold = fold
            model_paths[fold] = train_fold()[0]
        eval_kfold(model_paths)

    else:
        train_fold()
