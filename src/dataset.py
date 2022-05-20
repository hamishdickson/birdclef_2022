import numpy as np
import pandas as pd
import torch

from .utils.general import AUGTIMER, LOADTIMER
from .utils.transforms import (
    Compose,
    GaussianNoise,
    NoiseInjection,
    Normalize,
    OneOf,
    PinkNoise,
    RandomVolume,
    blend_audio,
    crop_audio_center,
    crop_or_pad,
    cvt_audio_to_array,
    cvt_audio_to_array_v2,
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
        split_audio_root=None,
        label_smoothing=0.0,
        bg_blend_chance=0.0,
        bg_blend_alpha=(0.3, 0.6),
        weighted_by_rating=False,
        sampling="random",
    ):
        super().__init__(df, sr, duration, mode=mode)
        self.labels_df = labels_df
        self.target_columns = target_columns
        self.split_audio_root = split_audio_root
        self.label_smoothing = label_smoothing if mode == "train" else 0.0
        self.bg_blend_chance = bg_blend_chance
        self.bg_blend_alpha = bg_blend_alpha
        self._loaded_audio = None
        if self.mode == "train" and self.bg_blend_chance > 0:
            print("Creating binary df for augmentations...")
            self.binary_df = pd.read_csv("data/train_backgrounds.csv")
            self.binary_df = self.binary_df[self.binary_df.label == 0]
            print(f"Removed positive labels from binary df, new shape {len(self.binary_df)}")
        self.df["weight"] = 1.0
        if weighted_by_rating:
            self.df["weight"] = self.df["rating"] / self.df["rating"].max()
        assert sampling in [
            "random",
            "birdcall",
        ], "Audio clip sampling must be one of ['random', 'birdcall']"
        self.sampling = sampling

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]

        wav_path = sample["file_path"]
        labels = sample["new_target"]
        weight = float(sample["weight"])
        is_scored = sample["is_scored"]

        with LOADTIMER:
            if self.sampling == "birdcall":
                y = cvt_audio_to_array(
                    wav_path,
                    labels_df=self.labels_df,
                    target_sr=self.sr,
                    duration=self.duration,
                    use_highest=self.mode != "train",
                    split_audio_root=self.split_audio_root,
                )
            elif self.sampling == "random":
                y = cvt_audio_to_array_v2(
                    wav_path,
                    labels_df=self.labels_df,
                    target_sr=self.sr,
                    duration=self.duration,
                    use_highest=self.mode != "train",
                    split_audio_root=self.split_audio_root,
                )

        if len(y) > 0 and self.wave_transforms:
            with AUGTIMER:
                y = self.wave_transforms(y, sr=self.sr)

        y = crop_or_pad(
            y,
            self.duration * self.sr,
            sr=self.sr,
            train=self.mode == "train",
            probs=None,
        )

        if self.mode == "train" and np.random.random() < self.bg_blend_chance:
            bin_path = self.binary_df.sample(n=1).iloc[0]["filepath"]

            # Load from cache
            bin_y = load_audio(bin_path, target_sr=self.sr)
            bin_y = crop_or_pad(
                bin_y,
                self.duration * self.sr,
                sr=self.sr,
                train=self.mode == "train",
                probs=None,
            )
            # import soundfile as sf

            # name = f"{str(np.random.random())}.wav"
            # sf.write(name, y, samplerate=self.sr)
            y = blend_audio(y, bin_y, alpha=self.bg_blend_alpha)
            # sf.write(name.replace(".wav", "_bend.wav"), y, samplerate=self.sr)

        y = torch.from_numpy(y).float()

        targets = np.ones(len(self.target_columns), dtype=float) * self.label_smoothing
        for ebird_code in labels.split():
            targets[self.target_columns.index(ebird_code)] = 1.0 - self.label_smoothing

        return {
            "audio": y,
            "targets": targets,
            "weights": weight,
            "is_scored": is_scored,
        }
