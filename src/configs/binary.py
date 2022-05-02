import numpy as np
import torch


class CFG:
    # List[Tuple[audio_root, csv_path]]
    train_data = [
        (
            "/media/nvme/Datasets/bird/ext/birdvox/wav",
            "/media/nvme/Datasets/bird/ext/birdvox/BirdVoxDCASE20k_csvpublic.csv",
        ),
        (
            "/media/nvme/Datasets/bird/ext/ff1010/raw/ff1010bird_wav/wav",
            "/media/nvme/Datasets/bird/ext/ff1010/raw/ff1010bird_wav/ff1010bird_metadata_2018.csv",
        ),
        (
            "/media/nvme/Datasets/bird/ext/warblrb10k_public/wav",
            "/media/nvme/Datasets/bird/ext/warblrb10k_public/warblrb10k_public_metadata_2018.csv",
        ),
    ]

    val_data = (
        "/media/nvme/Datasets/bird/2021/train_soundscapes",
        "/media/nvme/Datasets/bird/2021/train_soundscape_labels.csv",
    )

    seed = 71
    epochs = 23
    cutmix_and_mixup_epochs = 18
    fold = 0  # [0, 1, 2, 3, 4]
    dropout = 0.1
    LR = 2 * 1e-3
    ETA_MIN = 1e-6
    WEIGHT_DECAY = 1e-6
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
