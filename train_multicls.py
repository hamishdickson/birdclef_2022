import ast
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler

from src.configs.multicls import CFG
from src.dataset import WaveformDataset
from src.engine import Trainer
from src.utils.general import set_seed
from src.utils.metrics import MetricMeter


def create_dataset(df, labels_df, mode, batch_size, nb_workers, shuffle):
    dataset = WaveformDataset(
        df=df,
        labels_df=labels_df,
        sr=CFG.sample_rate,
        duration=CFG.period,
        target_columns=CFG.target_columns,
        mode=mode,
        split_audio_root=CFG.split_audios_path,
        label_smoothing=CFG.label_smoothing,
        bg_blend_chance=CFG.bg_blend_chance,
        weighted_by_rating=CFG.weighted_by_rating,
        sampling=CFG.sampling,
    )
    sampler = None
    if mode == "train" and CFG.class_count_sensitive_sampler:
        sampler = WeightedRandomSampler(df["sample_weight"], len(df))
        print("Used class-count sensitive sampler")
        shuffle = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nb_workers,
        pin_memory=True,
        shuffle=shuffle,
        sampler=sampler,
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
    df["sample_weight"] = df.groupby("primary_label")["primary_label"].transform("count")
    df["sample_weight"] = 1 / np.log1p(df["sample_weight"])
    assert len(labels_df) > 0, "Train soundscapes are empty"
    return df, labels_df


def train_fold(trainer: Trainer, df: pd.DataFrame, labels_df: pd.DataFrame):
    # make sure each fold has different seed set, dataset split seed set separately
    set_seed(CFG.seed + CFG.fold)
    train_df = df[df.kfold != CFG.fold].reset_index(drop=True)
    val_df = df[df.kfold == CFG.fold].reset_index(drop=True)
    _, train_dataloader = create_dataset(
        train_df, labels_df, "train", CFG.train_bs, CFG.nb_workers, shuffle=True
    )
    _, valid_dataloader = create_dataset(
        val_df, labels_df, "val", CFG.valid_bs, CFG.nb_workers, shuffle=False
    )
    return trainer.train(train_dataloader, valid_dataloader)


def main(args):
    trainer = Trainer(CFG, torch.device(args.device))
    df, labels_df = create_df()

    if args.test_only:
        model_paths = [
            str(p)
            for p in Path(trainer._output_dir).glob("*.bin")
            if CFG.base_model_name in str(p)
        ]
        score = MetricMeter(optimise_per_bird=True, ranges=np.arange(0.01, 0.31, 0.01))
        for path in model_paths:
            fold = int(path.split("/")[-1][5])
            print(f"Processing fold {fold}...")
            val_df = df[df.kfold == fold].reset_index(drop=True)
            _, valid_dataloader = create_dataset(
                val_df, labels_df, "val", CFG.valid_bs, CFG.nb_workers, shuffle=False
            )
            score = trainer.validate(valid_dataloader, path, score_meter=score)
        print(score.avg)
        return

    if args.multi_fold:
        for fold in range(args.nb_folds):
            CFG.fold = fold
            train_fold(trainer, df, labels_df)
    else:
        assert CFG.fold == args.fold, "config. fold must be equal to args fold"
        train_fold(trainer, df, labels_df)
    return


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--multi-fold", action="store_true")
    parser.add_argument("--nb-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--exp-name", type=str, default="", help="experiment name")
    parser.add_argument("--nb-workers", type=int, default=8, help="number of dataloader workers")
    parser.add_argument("--ls", type=float, default=0.0, help="label smoothing")
    parser.add_argument("--bg-blend", type=float, default=0.0, help="add background audio")
    parser.add_argument("--train-bs", type=int, default=16, help="train batch size")
    parser.add_argument("--gd", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--base-model", type=str, help="timm backbone")
    parser.add_argument("--meta-model", type=str, help="meta model", default="TimmSED")
    parser.add_argument("--dp", type=float, default=0.0, help="drop path rate")
    parser.add_argument("--loss", type=str, default="FocalLoss", help="loss function")
    parser.add_argument(
        "--wbr",
        dest="wbr",
        help="Weigh samples by rating",
        action="store_true",
    )
    parser.add_argument(
        "--ccs-sampler",
        dest="ccs_sampler",
        help="apply class-count sensitive sampler",
        action="store_true",
    )
    parser.add_argument("--in-chans", type=int, help="number of channels", default=3)
    parser.add_argument("--mel", type=str, help="melspec. type", default="delta")
    parser.add_argument("--per", type=int, default=30, help="clip length in seconds")
    parser.add_argument("--sampling", type=str, default="random", help="clip sampling method")
    parser.add_argument("--test-only", dest="test_only", help="Run inference", action="store_true")
    parser.add_argument("--device", type=str, help="Set device", default="cuda")
    args = parser.parse_args()
    print(args)

    CFG.nb_workers = args.nb_workers
    CFG.fold = args.fold
    CFG.exp_name = args.exp_name
    CFG.label_smoothing = args.ls
    CFG.bg_blend_chance = args.bg_blend
    CFG.train_bs = args.train_bs
    CFG.grad_acc_steps = args.gd
    CFG.base_model_name = args.base_model
    CFG.meta_model_name = args.meta_model
    CFG.drop_path = args.dp
    CFG.loss_name = args.loss
    CFG.weighted_by_rating = args.wbr
    CFG.class_count_sensitive_sampler = args.ccs_sampler
    CFG.in_chans = args.in_chans
    CFG.melspec_type = args.mel
    CFG.period = args.per
    CFG.sampling = args.sampling

    main(args)
