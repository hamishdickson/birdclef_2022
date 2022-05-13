import numpy as np
import torch


class CFG:
    exp_name = ""
    output_dir = "../exp/binary/"

    # List[Tuple[audio_root, csv_path]]
    train_data = [
        (
            "/mnt/datastore/birdclef_2022/ext/birdvox/wav",
            "/mnt/datastore/birdclef_2022/ext/birdvox/BirdVoxDCASE20k_csvpublic.csv",
        ),
        (
            "/mnt/datastore/birdclef_2022/ext/ff1010/raw/ff1010bird_wav/wav",
            "/mnt/datastore/birdclef_2022/ext/ff1010/raw/ff1010bird_wav/ff1010bird_metadata_2018.csv",
        ),
        # (
        #     "/mnt/datastore/birdclef_2022/ext/warblrb10k_public/wav",
        #     "/mnt/datastore/birdclef_2022/ext/warblrb10k_public/warblrb10k_public_metadata_2018.csv",
        # ),
    ]

    val_data = (
        "/mnt/datastore/birdclef_2022/ext/2021/train_soundscapes",
        "data/train_soundscape_labels.csv",
    )

    seed = 71
    epochs = 23
    cutmix_and_mixup_epochs = 18
    fold = 0  # [0, 1, 2, 3, 4]
    dropout = 0.1
    LR = 2 * 1e-3
    ETA_MIN = 1e-6
    WEIGHT_DECAY = 1e-6
    mixup_alpha = 0.4
    train_bs = 32  # 32
    valid_bs = 32  # 64
    base_model_name = "tf_efficientnet_b0_ns"
    apex = True
    nb_workers = 6

    mean = (
        torch.from_numpy(np.array([0.485, 0.456, 0.406])).float()[None, :, None, None].cuda()
    )  # RGB
    std = (
        torch.from_numpy(np.array([0.229, 0.224, 0.225])).float()[None, :, None, None].cuda()
    )  # RG

    pretrained = True
    num_classes = 1

    period = 5
    n_mels = 224  # 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
