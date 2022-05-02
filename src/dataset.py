import numpy as np
import pandas as pd
import torch

from utils.general import AUGTIMER, LOADTIMER
from utils.transforms import (
    Compose,
    GaussianNoise,
    NoiseInjection,
    Normalize,
    OneOf,
    PinkNoise,
    RandomVolume,
    crop_audio_center,
    crop_or_pad,
    cvt_audio_to_array,
    load_audio,
)


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, sr: int, duration: float, mode="train"):
        self.df = df
        self.sr = sr
        self.duration = duration
        self.mode = mode

        if mode == "train":
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
        sample = self.df.iloc[idx]

        wav_path = sample["filepath"]
        label = sample["label"]

        with LOADTIMER:
            y = load_audio(wav_path, target_sr=self.sr)

        if self.mode == "train":
            # Train is just 10s files
            y = crop_audio_center(y, target_sr=self.sr, duration=self.duration)
        else:
            # Val uses train_soundscapes 2021
            end = int(sample["seconds"])
            start = end - 5
            y = y[start * self.sr : end * self.sr]

        if len(y) > 0 and self.wave_transforms:
            with AUGTIMER:
                y = self.wave_transforms(y, sr=self.sr)
        # print("after transform: ", y.shape)

        y = crop_or_pad(
            y,
            self.duration * self.sr,
            sr=self.sr,
            train=self.mode == "train",
            probs=None,
        )
        # print("after croppad: ", y.shape)
        y = torch.from_numpy(y).float()
        targets = torch.from_numpy(np.atleast_1d(label)).float()

        return {
            "audio": y,
            "targets": targets,
        }


class WaveformDataset(BinaryDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labels_df: pd.DataFrame,
        sr: int,
        duration: float,
        target_columns: list,
        mode="train",
    ):
        super().__init__(df, sr, duration, mode=mode)
        self.labels_df = labels_df
        self.target_columns = target_columns

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]

        wav_path = sample["file_path"]
        labels = sample["new_target"]

        with LOADTIMER:
            y = cvt_audio_to_array(
                wav_path,
                labels_df=self.labels_df,
                target_sr=self.sr,
                duration=self.duration,
                use_highest=self.mode != "train",
            )

        if len(y) > 0 and self.wave_transforms:
            with AUGTIMER:
                y = self.wave_transforms(y, sr=self.sr)
        #                 print("after transform: ", y.shape)

        y = crop_or_pad(
            y,
            self.duration * self.sr,
            sr=self.sr,
            train=self.mode == "train",
            probs=None,
        )
        #         print("after croppad: ", y.shape)
        y = torch.from_numpy(y).float()

        targets = np.zeros(len(self.target_columns), dtype=float)
        for ebird_code in labels.split():
            targets[self.target_columns.index(ebird_code)] = 1.0

        return {
            "audio": y,
            "targets": targets,
        }
