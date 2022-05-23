import numpy as np
import torch

from configs.multicls import CFG
from model import TimmSED
from train_multicls import create_dataset, create_df
from utils.metrics import MetricMeter

from tqdm import tqdm


def eval_kfold(model_paths):

    df, labels_df = create_df()

    best_score = 0.0
    best_alpha = 0.0
    best_beta = 0.0
    best_gamma = 0.0

    best_avgs = None    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score_meter = MetricMeter(optimise_per_bird=True, ranges=np.arange(0.01, 0.31, 0.01))

    y_true = []
    y_pred_1 = []
    y_pred_2 = []
    y_pred_3 = []
    y_pred_4 = []

    for fold, path in model_paths.items():
        print(f"Processing fold {fold}...")
        val_df = df[df.kfold == fold].reset_index(drop=True)

        valid_dateset, valid_dataloader = create_dataset(
                val_df, labels_df, "val", CFG.valid_bs, CFG.nb_workers, shuffle=False
            )
        model_1 = TimmSED(
                    cfg=CFG,
                    base_model_name="tf_efficientnet_b6_ns",
                    pretrained=CFG.pretrained,
                    num_classes=CFG.num_classes,
                ).to(device)

        model_2 = TimmSED(
                    cfg=CFG,
                    base_model_name="tf_efficientnet_b5_ns",
                    pretrained=CFG.pretrained,
                    num_classes=CFG.num_classes,
                ).to(device)

        model_3 = TimmSED(
                    cfg=CFG,
                    base_model_name="tf_efficientnet_b4_ns",
                    pretrained=CFG.pretrained,
                    num_classes=CFG.num_classes,
                ).to(device)

        model_4 = TimmSED(
                    cfg=CFG,
                    base_model_name="tf_efficientnet_b3_ns",
                    pretrained=CFG.pretrained,
                    num_classes=CFG.num_classes,
                ).to(device)
        model_1.load_state_dict(torch.load(path[3]), strict=True)
        model_2.load_state_dict(torch.load(path[2]), strict=True)
        model_3.load_state_dict(torch.load(path[1]), strict=True)
        model_4.load_state_dict(torch.load(path[0]), strict=True)

        model_1.eval()
        model_2.eval()
        model_3.eval()
        model_4.eval()

        tk0 = tqdm(valid_dataloader, total=len(valid_dataloader))
        with torch.no_grad():
            for data in tk0:
                inputs = data["audio"].to(device)
                targets = data["targets"].to(device)
                outputs_1 = model_1(inputs)
                outputs_2 = model_2(inputs)
                outputs_3 = model_3(inputs)
                outputs_4 = model_4(inputs)

                y_true.extend(targets.cpu().detach().numpy().tolist())
                y_pred_1.extend(outputs_1["clipwise_output"].cpu().detach().numpy().tolist())
                y_pred_2.extend(outputs_2["clipwise_output"].cpu().detach().numpy().tolist())
                y_pred_3.extend(outputs_3["clipwise_output"].cpu().detach().numpy().tolist())
                y_pred_4.extend(outputs_4["clipwise_output"].cpu().detach().numpy().tolist())

    score_meter.y_true = y_true
    for alpha in np.arange(0.1, 0.5, 0.01):
        outputs_cw = alpha * np.array(y_pred_2) + (1 - alpha) * np.array(y_pred_1)
        score_meter.y_pred = outputs_cw

        ave = score_meter.avg
        print(alpha, ave)

        if ave['masked_optimised_global_score'] > best_score:
            print(f"new best score {ave['masked_optimised_global_score']}, alpha {alpha}")
            best_alpha = alpha
            best_score = ave['masked_optimised_global_score']
            best_avgs = ave

    print(best_alpha, best_score, best_avgs)
    print()
    print("**************************************************************")
    print()
    print("start beta search")

    for beta in np.arange(0.01, 0.25, 0.01):
        outputs_cw = beta * np.array(y_pred_3) + (1 - beta) * (best_alpha * np.array(y_pred_2) + (1 - best_alpha) * np.array(y_pred_1))
        score_meter.y_pred = outputs_cw

        ave = score_meter.avg
        print(beta, ave)

        if ave['masked_optimised_global_score'] > best_score:
            print(f"new best score {ave['masked_optimised_global_score']}, alpha {best_alpha}, beta {beta}")
            best_beta = beta
            best_score = ave['masked_optimised_global_score']
            best_avgs = ave

    print(best_alpha, best_beta, best_score, best_avgs)

    print()
    print("**************************************************************")
    print()
    print("start gamma search")

    for gamma in np.arange(0.01, 0.25, 0.01):
        outputs_cw = gamma * np.array(y_pred_4) + (1 - gamma) * (best_beta * np.array(y_pred_3) + (1 - best_beta) * (best_alpha * np.array(y_pred_2) + (1 - best_alpha) * np.array(y_pred_1)))
        score_meter.y_pred = outputs_cw

        ave = score_meter.avg
        print(gamma, ave)

        if ave['masked_optimised_global_score'] > best_score:
            print(f"new best score {ave['masked_optimised_global_score']}, alpha {best_alpha}, beta {best_beta}, gamma {gamma}")
            best_gamma = gamma
            best_score = ave['masked_optimised_global_score']
            best_avgs = ave

    print(best_alpha, best_beta, best_gamma, best_score, best_avgs)



if __name__ == "__main__":
    model_paths = {
        0: [
            "/home/hamsh/Workspace/exp/ensemble/fold-0-tf_efficientnet_b3_ns-multichannel_power_energy_pcen-epoch-21-f1-0.931-0.915.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-0-tf_efficientnet_b4_ns-multichannel_power_energy_pcen-epoch-12-f1-0.902-0.898.bin", 
            "/home/hamsh/Workspace/exp/ensemble/fold-0-tf_efficientnet_b5_ns-multichannel_power_energy_pcen-epoch-21-f1-0.931-0.910.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-0-tf_efficientnet_b6_ns-multichannel_power_energy_pcen-epoch-18-f1-0.925-0.906.bin"
        ],
        1: [
            "/home/hamsh/Workspace/exp/ensemble/fold-1-tf_efficientnet_b3_ns-multichannel_power_energy_pcen-epoch-22-f1-0.901-0.909.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-1-tf_efficientnet_b4_ns-multichannel_power_energy_pcen-epoch-12-f1-0.874-0.890.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-1-tf_efficientnet_b5_ns-multichannel_power_energy_pcen-epoch-13-f1-0.899-0.902.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-1-tf_efficientnet_b6_ns-multichannel_power_energy_pcen-epoch-18-f1-0.874-0.889.bin"
            ],
        2: [
            "/home/hamsh/Workspace/exp/ensemble/fold-2-tf_efficientnet_b3_ns-multichannel_power_energy_pcen-epoch-13-f1-0.934-0.894.bin", 
            "/home/hamsh/Workspace/exp/ensemble/fold-2-tf_efficientnet_b4_ns-multichannel_power_energy_pcen-epoch-18-f1-0.949-0.914.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-2-tf_efficientnet_b5_ns-multichannel_power_energy_pcen-epoch-18-f1-0.963-0.912.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-2-tf_efficientnet_b6_ns-multichannel_power_energy_pcen-epoch-18-f1-0.948-0.904.bin"
            ],
        3: [
            "/home/hamsh/Workspace/exp/ensemble/fold-3-tf_efficientnet_b3_ns-multichannel_power_energy_pcen-epoch-21-f1-0.950-0.914.bin", 
            "/home/hamsh/Workspace/exp/ensemble/fold-3-tf_efficientnet_b4_ns-multichannel_power_energy_pcen-epoch-20-f1-0.947-0.905.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-3-tf_efficientnet_b5_ns-multichannel_power_energy_pcen-epoch-19-f1-0.948-0.916.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-3-tf_efficientnet_b6_ns-multichannel_power_energy_pcen-epoch-20-f1-0.934-0.901.bin"
            ],
        4: [
            "/home/hamsh/Workspace/exp/ensemble/fold-4-tf_efficientnet_b3_ns-multichannel_power_energy_pcen-epoch-18-f1-0.874-0.913.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-4-tf_efficientnet_b4_ns-multichannel_power_energy_pcen-epoch-22-f1-0.911-0.908.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-4-tf_efficientnet_b5_ns-multichannel_power_energy_pcen-epoch-21-f1-0.914-0.915.bin",
            "/home/hamsh/Workspace/exp/ensemble/fold-4-tf_efficientnet_b6_ns-multichannel_power_energy_pcen-epoch-19-f1-0.905-0.909.bin"
        ]
    }


    eval_kfold(model_paths)
