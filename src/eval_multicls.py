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

    s = score.avg
    print(s)
    return s


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

    # fixed_norm_first_channel_drop25_blend_chance80_alpha30 {'f1_at_best': (0.11, 0.8756303435061916), 'masked_best_ths_per_bird': {'akiapo': 0.08, 'aniani': 0.12, 'apapan': 0.17, 'barpet': 0.14, 'crehon': 0.06999999999999999, 'elepai': 0.12, 'ercfra': 0.04, 'hawama': 0.13, 'hawcre': 0.14, 'hawgoo': 0.05, 'hawhaw': 0.05, 'hawpet1': 0.04, 'houfin': 0.23, 'iiwi': 0.17, 'jabwar': 0.14, 'maupar': 0.02, 'omao': 0.15000000000000002, 'puaioh': 0.04, 'skylar': 0.22, 'warwhe1': 0.13, 'yefcan': 0.09}, 'masked_best_score_per_bird': {'akiapo': 0.9834512976070104, 'aniani': 0.959185369908562, 'apapan': 0.972117063896245, 'barpet': 0.9311091640268698, 'crehon': 0.993973063973064, 'elepai': 0.9611665393315656, 'ercfra': 0.8607930306704388, 'hawama': 0.9821126665314127, 'hawcre': 0.9937622226717917, 'hawgoo': 0.9603800282999797, 'hawhaw': 0.811277976070218, 'hawpet1': 0.794307136283027, 'houfin': 0.910577427858529, 'iiwi': 0.966464947537973, 'jabwar': 0.9266889978079358, 'maupar': 0.9168742845599622, 'omao': 0.9645706635312251, 'puaioh': 0.9599299616135767, 'skylar': 0.9597248330973875, 'warwhe1': 0.9283129110881105, 'yefcan': 0.927265160628101}, 'masked_optimised_global_score': 0.936383083190142}

    # fixed_norm_first_channel_drop10_blend_chance80_alpha30 {'f1_at_best': (0.08, 0.8855088871993559), 'masked_best_ths_per_bird': {'akiapo': 0.09, 'aniani': 0.09, 'apapan': 0.11, 'barpet': 0.09, 'crehon': 0.08, 'elepai': 0.09999999999999999, 'ercfra': 0.04, 'hawama': 0.11, 'hawcre': 0.15000000000000002, 'hawgoo': 0.04, 'hawhaw': 0.03, 'hawpet1': 0.03, 'houfin': 0.21000000000000002, 'iiwi': 0.08, 'jabwar': 0.08, 'maupar': 0.02, 'omao': 0.13, 'puaioh': 0.04, 'skylar': 0.22, 'warwhe1': 0.12, 'yefcan': 0.06999999999999999}, 'masked_best_score_per_bird': {'akiapo': 0.9900910010111224, 'aniani': 0.9570285996719912, 'apapan': 0.9542989067698784, 'barpet': 0.9954168632472872, 'crehon': 0.9972053872053872, 'elepai': 0.9622114361901337, 'ercfra': 0.7968251829898064, 'hawama': 0.968756860235862, 'hawcre': 0.9956504147278981, 'hawgoo': 0.9575500303214068, 'hawhaw': 0.9648461175836757, 'hawpet1': 0.7892226188071025, 'houfin': 0.9117772987067432, 'iiwi': 0.9707687907389845, 'jabwar': 0.9178268392404128, 'maupar': 0.9505757188068144, 'omao': 0.9547724087733079, 'puaioh': 0.8164635104496374, 'skylar': 0.9607802713247258, 'warwhe1': 0.9174184501352209, 'yefcan': 0.9439985412150855}, 'masked_optimised_global_score': 0.9368326308644038}

    # fixed_norm_first_channel_drop25_blend_chance80_alpha30 {'f1_at_best': (0.060000000000000005, 0.8976078315237482), 'masked_best_ths_per_bird': {'akiapo': 0.09999999999999999, 'aniani': 0.03, 'apapan': 0.11, 'barpet': 0.04, 'crehon': 0.03, 'elepai': 0.03, 'ercfra': 0.11, 'hawama': 0.08, 'hawcre': 0.21000000000000002, 'hawgoo': 0.04, 'hawhaw': 0.01, 'hawpet1': 0.03, 'houfin': 0.18000000000000002, 'iiwi': 0.08, 'jabwar': 0.08, 'maupar': 0.13, 'omao': 0.060000000000000005, 'puaioh': 0.02, 'skylar': 0.15000000000000002, 'warwhe1': 0.09, 'yefcan': 0.05}, 'masked_best_score_per_bird': {'akiapo': 0.9935288169868555, 'aniani': 0.9702096111073667, 'apapan': 0.9765861343111293, 'barpet': 0.9519736694300285, 'crehon': 0.986902356902357, 'elepai': 0.960799514628556, 'ercfra': 0.9153868606583142, 'hawama': 0.9814363968350579, 'hawcre': 0.9980106547980309, 'hawgoo': 0.984502392022101, 'hawhaw': 0.8740992659438346, 'hawpet1': 0.9815812512627113, 'houfin': 0.916903734764468, 'iiwi': 0.982624305649641, 'jabwar': 0.9334052732691314, 'maupar': 0.5, 'omao': 0.9817027884680305, 'puaioh': 0.9599299616135767, 'skylar': 0.9583906436548641, 'warwhe1': 0.9314357895674046, 'yefcan': 0.9460362323742411}, 'masked_optimised_global_score': 0.9374021740117952}

    model_paths = {
        0: "/media/hdd/Kaggle/bird/exp/multiclass/05-12-22-06:27:27-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler/fold-0-tf_efficientnet_b5_ns-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler-epoch-11-f1-0.932-0.913.bin",
        1: "/media/hdd/Kaggle/bird/exp/multiclass/05-12-22-07:10:20-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler/fold-1-tf_efficientnet_b5_ns-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler-epoch-21-f1-0.867-0.912.bin",
        2: "/media/hdd/Kaggle/bird/exp/multiclass/05-12-22-07:53:08-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler/fold-2-tf_efficientnet_b5_ns-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler-epoch-22-f1-0.942-0.900.bin",
        3: "/media/hdd/Kaggle/bird/exp/multiclass/05-12-22-08:36:29-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler/fold-3-tf_efficientnet_b5_ns-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler-epoch-20-f1-0.918-0.903.bin",
        4: "/media/hdd/Kaggle/bird/exp/multiclass/05-12-22-09:20:01-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler/fold-4-tf_efficientnet_b5_ns-fixed_norm_first_channel_drop5_blend_chance80_alpha30-60_sampler-epoch-19-f1-0.911-0.919.bin",
    }  # b5 no_blen lb 81 fixed_norm_first_channel_drop5_blend_chance80_alpha30 {'f1_at_best': (0.09999999999999999, 0.8984939527644512), 'masked_best_ths_per_bird': {'akiapo': 0.06999999999999999, 'aniani': 0.08, 'apapan': 0.17, 'barpet': 0.05, 'crehon': 0.060000000000000005, 'elepai': 0.060000000000000005, 'ercfra': 0.06999999999999999, 'hawama': 0.12, 'hawcre': 0.08, 'hawgoo': 0.04, 'hawhaw': 0.05, 'hawpet1': 0.06999999999999999, 'houfin': 0.19, 'iiwi': 0.13, 'jabwar': 0.09, 'maupar': 0.29000000000000004, 'omao': 0.08, 'puaioh': 0.04, 'skylar': 0.19, 'warwhe1': 0.09999999999999999, 'yefcan': 0.12}, 'masked_best_score_per_bird': {'akiapo': 0.9863835524098417, 'aniani': 0.9901934353305925, 'apapan': 0.9497127724587493, 'barpet': 0.9478623261665655, 'crehon': 0.993030303030303, 'elepai': 0.9727315626263988, 'ercfra': 0.9929273878485787, 'hawama': 0.96113217951319, 'hawcre': 0.9913008294557961, 'hawgoo': 0.9706556161983694, 'hawhaw': 0.6541405706332637, 'hawpet1': 0.8248142411385727, 'houfin': 0.919078823791313, 'iiwi': 0.9768647729154436, 'jabwar': 0.9222371370163296, 'maupar': 0.49983166116759814, 'omao': 0.9679210079522307, 'puaioh': 0.9873392147619369, 'skylar': 0.9515206252691457, 'warwhe1': 0.9228139294180495, 'yefcan': 0.9576545522814592}, 'masked_optimised_global_score': 0.920959357208749}

    # b3 fixed_norm_first_channel_drop5_blend_chance80_alpha30 {'f1_at_best': (0.13, 0.8959096739418515), 'masked_best_ths_per_bird': {'akiapo': 0.15000000000000002, 'aniani': 0.08, 'apapan': 0.14, 'barpet': 0.11, 'crehon': 0.16, 'elepai': 0.09, 'ercfra': 0.04, 'hawama': 0.14, 'hawcre': 0.15000000000000002, 'hawgoo': 0.08, 'hawhaw': 0.01, 'hawpet1': 0.09999999999999999, 'houfin': 0.22, 'iiwi': 0.13, 'jabwar': 0.12, 'maupar': 0.3, 'omao': 0.11, 'puaioh': 0.09, 'skylar': 0.26, 'warwhe1': 0.13, 'yefcan': 0.15000000000000002}, 'masked_best_score_per_bird': {'akiapo': 0.9926862150320188, 'aniani': 0.9789377906584888, 'apapan': 0.9650766619134923, 'barpet': 0.9896205432365033, 'crehon': 0.9986531986531986, 'elepai': 0.9729000943777808, 'ercfra': 0.9383672369661862, 'hawama': 0.9726620576285823, 'hawcre': 0.9728997528344872, 'hawgoo': 0.9763829930597668, 'hawhaw': 0.7125058926527039, 'hawpet1': 0.8234336767908053, 'houfin': 0.9126355690347157, 'iiwi': 0.96385840521459, 'jabwar': 0.9158702987611143, 'maupar': 0.4998989967005589, 'omao': 0.9522149819432162, 'puaioh': 0.9935012458751431, 'skylar': 0.9586625614683819, 'warwhe1': 0.9172988102790113, 'yefcan': 0.934799274921154}, 'masked_optimised_global_score': 0.921088869428662}

    eval_kfold(model_paths)
