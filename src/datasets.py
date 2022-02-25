import tqdm
import random
import pandas as pd
import numpy as np
from skimage.transform import resize
import librosa
import torch.utils.data as torchdata

import cv2
import torch

import numpy as np

from PIL import Image
from PIL import ImageFile

def spec_to_image(spec, eps=1e-10):
    """
    dumb scaling, we can probably do better
    """
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
    # spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


class BirdDataset:
    """
    we will have to write something much more insane for the validation
    """
    def __init__(self, df, augmentations):
        self.primary_label = df.primary_label.values
        self.secondary_labels = df.secondary_labels.values
        self.bird_time = df.time.values

        self.augmentations = augmentations
        self.grayscale = False

        # TODO probably all needs changing
        self.fft = 2048
        self.hop = 512
        self.sr = 32000,
        self.fmin=84,
        self.fmax=15056,
        self.power=2,
        self.n_mels=224

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        wav, sr = librosa.load('input/train_audio/' + self.primary_label[item] + '.ogg', sr=None)

        wav_slice = wav ## TODO here slice the wave about the bird time

        mel_spec = librosa.feature.melspectrogram(
            wav_slice,
            n_fft=self.fft, 
            hop_length=self.hop, 
            sr=self.sr, 
            fmin=self.fmin, 
            fmax=self.fmax, 
            power=self.power, 
            # n_mels=self.n_mels
        )

        targets = self.targets[item]

        image = mel_spec
        image = spec_to_image(image)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image_tensor = torch.tensor(image)
        if self.grayscale:
            image_tensor = image_tensor.unsqueeze(0)
        return {
            "image": image_tensor,
            "targets": torch.tensor(targets),
        }
