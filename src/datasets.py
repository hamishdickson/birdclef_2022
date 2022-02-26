import os
import tqdm
import soundfile as sf
import random
import pandas as pd
import numpy as np
from skimage.transform import resize
import librosa
import torch.utils.data as torchdata

from torch.utils.data import DataLoader, Dataset

import colorednoise as cn
import cv2
import torch

import numpy as np

from PIL import Image
from PIL import ImageFile


target_columns = [
        "afrsil1",
        "akekee",
        "akepa1",
        "akiapo",
        "akikik",
        "amewig",
        "aniani",
        "apapan",
        "arcter",
        "barpet",
        "bcnher",
        "belkin1",
        "bkbplo",
        "bknsti",
        "bkwpet",
        "blkfra",
        "blknod",
        "bongul",
        "brant",
        "brnboo",
        "brnnod",
        "brnowl",
        "brtcur",
        "bubsan",
        "buffle",
        "bulpet",
        "burpar",
        "buwtea",
        "cacgoo1",
        "calqua",
        "cangoo",
        "canvas",
        "caster1",
        "categr",
        "chbsan",
        "chemun",
        "chukar",
        "cintea",
        "comgal1",
        "commyn",
        "compea",
        "comsan",
        "comwax",
        "coopet",
        "crehon",
        "dunlin",
        "elepai",
        "ercfra",
        "eurwig",
        "fragul",
        "gadwal",
        "gamqua",
        "glwgul",
        "gnwtea",
        "golphe",
        "grbher3",
        "grefri",
        "gresca",
        "gryfra",
        "gwfgoo",
        "hawama",
        "hawcoo",
        "hawcre",
        "hawgoo",
        "hawhaw",
        "hawpet1",
        "hoomer",
        "houfin",
        "houspa",
        "hudgod",
        "iiwi",
        "incter1",
        "jabwar",
        "japqua",
        "kalphe",
        "kauama",
        "laugul",
        "layalb",
        "lcspet",
        "leasan",
        "leater1",
        "lessca",
        "lesyel",
        "lobdow",
        "lotjae",
        "madpet",
        "magpet1",
        "mallar3",
        "masboo",
        "mauala",
        "maupar",
        "merlin",
        "mitpar",
        "moudov",
        "norcar",
        "norhar2",
        "normoc",
        "norpin",
        "norsho",
        "nutman",
        "oahama",
        "omao",
        "osprey",
        "pagplo",
        "palila",
        "parjae",
        "pecsan",
        "peflov",
        "perfal",
        "pibgre",
        "pomjae",
        "puaioh",
        "reccar",
        "redava",
        "redjun",
        "redpha1",
        "refboo",
        "rempar",
        "rettro",
        "ribgul",
        "rinduc",
        "rinphe",
        "rocpig",
        "rorpar",
        "rudtur",
        "ruff",
        "saffin",
        "sander",
        "semplo",
        "sheowl",
        "shtsan",
        "skylar",
        "snogoo",
        "sooshe",
        "sooter1",
        "sopsku1",
        "sora",
        "spodov",
        "sposan",
        "towsol",
        "wantat1",
        "warwhe1",
        "wesmea",
        "wessan",
        "wetshe",
        "whfibi",
        "whiter",
        "whttro",
        "wiltur",
        "yebcar",
        "yefcan",
        "zebdov",
]
bird2id = {b: i for i, b in enumerate(target_columns)}
id2bird = {i: b for i, b in enumerate(target_columns)}
scored_birds = ["akiapo", "aniani", "apapan", "barpet", "crehon", "elepai", "ercfra", "hawama", "hawcre", "hawgoo", "hawhaw", "hawpet1", "houfin", "iiwi", "jabwar", "maupar", "omao", "puaioh", "skylar", "warwhe1", "yefcan"]



class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class OneOf(Compose):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class NewNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5):
        super().__init__(always_apply, p)
        self.max_range = max_range

    def apply(self, y: np.ndarray, sr, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1):
        super().__init__(always_apply, p)
        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs


def drop_stripes(image: np.ndarray, dim: int, drop_width: int, stripes_num: int):
    total_width = image.shape[dim]
    lowest_value = image.min()
    for _ in range(stripes_num):
        distance = np.random.randint(low=0, high=drop_width, size=(1,))[0]
        begin = np.random.randint(low=0, high=total_width - distance, size=(1,))[0]

        if dim == 0:
            image[begin : begin + distance] = lowest_value
        elif dim == 1:
            image[:, begin + distance] = lowest_value
        elif dim == 2:
            image[:, :, begin + distance] = lowest_value
    return image


def load_wave_and_crop(filename, period, start=None):
    # https://www.kaggle.com/c/birdclef-2022/discussion/308579
    waveform_orig, sample_rate = sf.read(filename, always_2d=True)
    waveform_orig = waveform_orig[:, 0]
    wave_len = len(waveform_orig)
    waveform = np.concatenate([waveform_orig, waveform_orig, waveform_orig])
    while len(waveform) < (period * sample_rate * 3):
        waveform = np.concatenate([waveform, waveform_orig])
    if start is not None:
        start = start - (period - 5) / 2 * sample_rate
        while start < 0:
            start += wave_len
        start = int(start)
        # start = int(start * sample_rate) + wave_len
    else:
        start = np.random.randint(wave_len)

    waveform_seg = waveform[start : start + int(period * sample_rate)]
    return waveform_orig, waveform_seg, sample_rate, start

class BirdClef2022Dataset(Dataset):
    def __init__(
        self,
        df,
        data_path: str = "input/train_audio",
        period: float = 15.0,
        secondary_coef: float = 1.0,
        smooth_label: float = 0.0,
        train: bool = True,
    ):

        self.df = df
        self.data_path = data_path
        self.filenames = df["filename"]
        self.primary_label = df["primary_label"]

        self.secondary_labels = (
            df["secondary_labels"]
            .map(
                lambda s: s.replace("[", "")
                .replace("]", "")
                .replace(",", "")
                .replace("'", "")
                .split(" ")
            )
            .values
        )
        self.secondary_coef = secondary_coef
        self.type = df["type"]
        self.period = period
        self.smooth_label = smooth_label + 1e-6
        if train:
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
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_path, self.filenames[idx]
        )
        if self.train:
            waveform, waveform_seg, sample_rate, start = load_wave_and_crop(
                filename, self.period
            )
        else:
            waveform, waveform_seg, sample_rate, start = load_wave_and_crop(
                filename, self.period, 0
            )

        waveform_seg = self.wave_transforms(waveform_seg, sr=sample_rate)

        waveform_seg = torch.Tensor(np.nan_to_num(waveform_seg))

        target = np.zeros(397, dtype=np.float32)
        primary_label = bird2id[self.primary_label[idx]]
        target[primary_label] = 1.0

        for s in self.secondary_labels[idx]:
            if s == "rocpig1":
                s = "rocpig"
            if s != "" and s in bird2id.keys():
                target[bird2id[s]] = self.secondary_coef

        target = torch.Tensor(target)
        return {
            "wave": waveform_seg,
            "target": (target > 0.01).float(),
            "loss_target": target * (1 - self.smooth_label)
            + self.smooth_label / target.size(-1),
        }

