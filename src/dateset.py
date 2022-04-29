import torch
import numpy as np
import pandas as pd

from config import CFG, AudioParams
from utils.transforms import Compose, OneOf, NoiseInjection, GaussianNoise, PinkNoise, RandomVolume, \
    Normalize, Audio_to_Array, crop_or_pad
from utils.general import LOADTIMER, AUGTIMER


class WaveformDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 labels_df: pd.DataFrame,
                 mode='train'):
        self.df = df
        self.labels_df = labels_df
        self.mode = mode

        if mode == 'train':
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            NoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=5, max_snr=20),
                            PinkNoise(p=1, min_snr=5, max_snr=20),
                        ],
                        p=0.2,
                    ),
                    RandomVolume(p=0.2, limit=4),
                    Normalize(p=1),
                ]
            )
        else:
            self.wave_transforms = Compose(
                [
                    Normalize(p=1),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]

        wav_path = sample["file_path"]
        labels = sample["new_target"]

        with LOADTIMER:
            y = Audio_to_Array(wav_path, labels_df=self.labels_df, use_highest=self.mode != "train")

        if len(y) > 0 and self.wave_transforms:
            with AUGTIMER:
                y = self.wave_transforms(y, sr=SR)
        #                 print("after transform: ", y.shape)

        y = crop_or_pad(
            y, AudioParams.duration * AudioParams.sr,
            sr=AudioParams.sr,
            train=self.mode == "train",
            probs=None)  # dont' do anythign since it is already 5s
        #         print("after croppad: ", y.shape)
        y = torch.from_numpy(y).float()

        targets = np.zeros(len(CFG.target_columns), dtype=float)
        for ebird_code in labels.split():
            targets[CFG.target_columns.index(ebird_code)] = 1.0

        return {
            "audio": y,
            "targets": targets,
        }
