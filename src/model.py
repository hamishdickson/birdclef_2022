import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchlibrosa.augmentation import SpecAugmentation

from nnAudio.features.cqt import CQT1992v2

from layers import AttBlockV2, init_bn, init_layer
from loss import loss_fn, mixup_criterion
from utils.general import cutmix, mixup
from utils.transforms import mono_to_color, multi_norm


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, cfg, pretrained=False, num_classes=24):
        super().__init__()
        self.cfg = cfg

        self.mel_trans_power = MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
            power=2.0
        )

        self.mel_trans_energy = MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
            power=1.0
        )


        self.mel_trans_energy15 = MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
            power=1.5
        )

        self.amp_db_tran = AmplitudeToDB()

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64 // 2, time_stripes_num=2, freq_drop_width=8 // 2, freq_stripes_num=2
        )

        self.bn0 = nn.BatchNorm2d(self.cfg.n_mels)

        self.encoder = timm.create_model(base_model_name, pretrained=pretrained)

        if hasattr(self.encoder, "fc"):
            in_features = self.encoder.fc.in_features
        else:
            in_features = self.encoder.classifier.in_features

        self.encoder.reset_classifier(0, "")

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def compute_melspec(self, y):
        return self.amp_db_tran(self.mel_trans_power(y)).float()

    def compute_melspec_multi_channel(self, y):
        return (self.amp_db_tran(self.mel_trans_power(y)).float(), self.amp_db_tran(self.mel_trans_energy15(y)).float(), self.amp_db_tran(self.mel_trans_energy(y)).float())

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, x, targets=None, do_mixup=False, weights=None):
        # (batch_size, len_audio)
        with autocast(enabled=False):
            with torch.no_grad():
                if self.cfg.multi_channel:
                    i, j, k = self.compute_melspec_multi_channel(x)
                    x = multi_norm(i, j, k).transpose(1, 3)  # (batch_size, 3, time_steps, mel_bins)
                else:
                    x = self.compute_melspec(x)
                    x = mono_to_color(x).transpose(1, 3)  # (batch_size, 3, time_steps, mel_bins)
                x = x - self.cfg.mean
                x = x / self.cfg.std

                if self.training and do_mixup:
                    if np.random.rand() < 0.5:
                        x, targets = mixup(x, targets, self.cfg.mixup_alpha, weights=weights)
                    else:
                        x, targets = cutmix(x, targets, self.cfg.mixup_alpha, weights=weights)

        x = x.transpose(1, 3)  # (batch_size, mel_bins, time_steps, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (batch_size, 3, time_steps, mel_bins)

        if self.training:
            if np.random.random() < 0.25:
                x = self.spec_augmenter(x)

        x = self.encoder(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)  # <- actually agg in the timestep axis

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)

        clipwise_output, logit = self.att_block(x)

        output_dict = {
            "clipwise_output": clipwise_output,  # (n_samples, n_class)
            "logit": logit,  # (n_samples, n_class)
        }

        loss = None
        if targets is not None:
            if do_mixup:
                loss = mixup_criterion(output_dict["logit"], targets)
            else:
                loss = loss_fn(output_dict["logit"], targets, weights=weights)
        output_dict["loss"] = loss
        return output_dict
