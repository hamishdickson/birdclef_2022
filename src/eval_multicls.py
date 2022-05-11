import numpy as np
import torch

from configs.multicls import CFG
from engine import valid_fn
from model import TimmSED
from train_multicls import create_dataset, create_df
from utils.metrics import MetricMeter


def eval_kfold(model_paths):
    df, labels_df = create_df()
    score = MetricMeter(optimise_per_bird=True, ranges=np.arange(0.01, 0.31, 0.01))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, path in model_paths.items():
        print(f"Processing fold {fold}...")
        val_df = df[df.kfold == fold].reset_index(drop=True)

        valid_dateset, valid_dataloader = create_dataset(
            val_df, labels_df, "val", CFG.valid_bs, CFG.nb_workers, shuffle=False
        )

        model = TimmSED(
            cfg=CFG,
            base_model_name=CFG.base_model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
        ).to(device)
        model.load_state_dict(torch.load(path), strict=True)

        score, _ = valid_fn(model, valid_dataloader, score_meter=score, device=device)
    print(score.avg)


if __name__ == "__main__":
    model_paths = {
        0: "/media/hdd/Kaggle/bird/exp/multiclass/05-06-22-09:51:41-weight1-alpha05/fold-0-tf_efficientnet_b3_ns-weight1-alpha05-epoch-22-f1-0.939-0.916.bin",
        1: "/media/hdd/Kaggle/bird/exp/multiclass/05-06-22-15:44:23-weight1-alpha05-seed71/fold-1-tf_efficientnet_b3_ns-weight1-alpha05-seed71-epoch-21-f1-0.896-0.913.bin",
        2: "/media/hdd/Kaggle/bird/exp/multiclass/05-06-22-16:15:31-weight1-alpha05-seed71/fold-2-tf_efficientnet_b3_ns-weight1-alpha05-seed71-epoch-22-f1-0.898-0.889.bin",
        3: "/media/hdd/Kaggle/bird/exp/multiclass/05-06-22-16:47:10-weight1-alpha05-seed71/fold-3-tf_efficientnet_b3_ns-weight1-alpha05-seed71-epoch-19-f1-0.938-0.911.bin",
        4: "/media/hdd/Kaggle/bird/exp/multiclass/05-06-22-17:19:53-weight1-alpha05-seed71/fold-4-tf_efficientnet_b3_ns-weight1-alpha05-seed71-epoch-22-f1-0.917-0.912.bin",
    }  # {'f1_at_best': (0.1, 0.8982563049870859), 'masked_f1_at_best': (0.05, 0.8936806066624238)
    # model_paths = {
    #     0: "/media/hdd/Kaggle/bird/exp/multiclass/05-09-22-08:04:29-weight1-alpha05-seed71-mixup60/fold-0-tf_efficientnet_b3_ns-weight1-alpha05-seed71-mixup60-epoch-21-f1-0.923-0.916.bin",
    #     1: "/media/hdd/Kaggle/bird/exp/multiclass/05-09-22-08:35:20-weight1-alpha05-seed71-mixup60/fold-1-tf_efficientnet_b3_ns-weight1-alpha05-seed71-mixup60-epoch-20-f1-0.901-0.916.bin",
    #     2: "/media/hdd/Kaggle/bird/exp/multiclass/05-09-22-10:35:29-weight1-alpha05-seed71-mixup60/fold-2-tf_efficientnet_b3_ns-weight1-alpha05-seed71-mixup60-epoch-20-f1-0.913-0.897.bin",
    #     3: "/media/hdd/Kaggle/bird/exp/multiclass/05-09-22-11:14:31-weight1-alpha05-seed71-mixup60/fold-3-tf_efficientnet_b3_ns-weight1-alpha05-seed71-mixup60-epoch-22-f1-0.948-0.906.bin",
    #     4: "/media/hdd/Kaggle/bird/exp/multiclass/05-09-22-11:50:14-weight1-alpha05-seed71-mixup60/fold-4-tf_efficientnet_b3_ns-weight1-alpha05-seed71-mixup60-epoch-21-f1-0.905-0.910.bin",
    # } {'f1_at_best': (0.1, 0.8990253005466433), 'masked_f1_at_best': (0.05, 0.8886466404790623)
    # model_paths = {
    #     0: "/media/hdd/Kaggle/bird/exp/multiclass/05-10-22-08:24:21-best-after-fixing-norm/fold-0-tf_efficientnet_b3_ns-best-after-fixing-norm-epoch-22-f1-0.896-0.908.bin",
    #     1: "/media/hdd/Kaggle/bird/exp/multiclass/05-10-22-08:54:29-best-after-fixing-norm/fold-1-tf_efficientnet_b3_ns-best-after-fixing-norm-epoch-20-f1-0.895-0.915.bin",
    #     2: "/media/hdd/Kaggle/bird/exp/multiclass/05-10-22-09:24:36-best-after-fixing-norm/fold-2-tf_efficientnet_b3_ns-best-after-fixing-norm-epoch-22-f1-0.929-0.906.bin",
    #     3: "/media/hdd/Kaggle/bird/exp/multiclass/05-10-22-09:54:51-best-after-fixing-norm/fold-3-tf_efficientnet_b3_ns-best-after-fixing-norm-epoch-22-f1-0.922-0.896.bin",
    #     4: "/media/hdd/Kaggle/bird/exp/multiclass/05-10-22-10:25:00-best-after-fixing-norm/fold-4-tf_efficientnet_b3_ns-best-after-fixing-norm-epoch-21-f1-0.914-0.905.bin",
    # }  # {'f1_at_best': (0.1, 0.8956286731237283), 'masked_f1_at_best': (0.05, 0.8726560635515258)}

    eval_kfold(model_paths)
