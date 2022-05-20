import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import model as models
from src.configs.multicls import CFG
from src.dataset import InferenceWaveformDataset
from src.engine import inference_fn
from train_multicls import args, create_df, get_model_paths, update_cfg


def create_dataset(df, mode, batch_size, nb_workers):
    dataset = InferenceWaveformDataset(
        df=df,
        sr=CFG.sample_rate,
        duration=CFG.period,
        target_columns=CFG.target_columns,
        mode=mode,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nb_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=InferenceWaveformDataset.collate_fn,
    )
    return dataset, dataloader


@torch.no_grad()
def inference_kfold(model_paths):
    df, _ = create_df()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = []
    probs = []
    for fold, path in model_paths.items():
        print(f"Processing fold {fold}... {path}")
        val_df = df[df.kfold == fold].reset_index(drop=True)

        _, valid_dataloader = create_dataset(val_df, "val", 4, CFG.nb_workers)

        model = getattr(models, CFG.meta_model_name)(
            cfg=CFG,
            base_model_name=CFG.base_model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
        ).to(device)
        model.load_state_dict(torch.load(path), strict=True)

        fold_meta, fold_probs = inference_fn(
            model, valid_dataloader, device=device, max_inf_size=200
        )
        metadata.extend(fold_meta)
        probs.extend(fold_probs)
    df = pd.DataFrame(metadata)
    probs = np.array(probs)
    df.columns = ["filename", "new_target"]
    print(probs.shape)
    print(df.shape)
    torch.save({"df": df, "probs": probs}, CFG.base_model_name + "_" + CFG.exp_name + ".pth")
    # df["probs"] = probs.tolist()
    # df.to_csv(f"data/{CFG.base_model_name}_pseudo.csv", index=False)


if __name__ == "__main__":
    CFG = update_cfg(CFG, args)
    model_paths = get_model_paths(
        Path(os.path.join(CFG.output_dir, CFG.exp_name)), CFG.base_model_name
    )
    inference_kfold(model_paths)
