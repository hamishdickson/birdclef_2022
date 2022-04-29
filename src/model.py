import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import timm

from torchlibrosa.augmentation import SpecAugmentation
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from layers import AttBlockV2, init_bn, init_layer
from loss import mixup_criterion, loss_fn
from utils.general import mixup, cutmix
from utils.transforms import mono_to_color

from config import CFG, AudioParams


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1):
        super().__init__()

        self.mel_trans = MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=CFG.n_fft,
            hop_length=CFG.hop_length,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
            n_mels=CFG.n_mels)
        self.amp_db_tran = AmplitudeToDB()

        self.spec_augmenter = SpecAugmentation(time_drop_width=64 // 2, time_stripes_num=2,
                                               freq_drop_width=8 // 2, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(CFG.n_mels)

        base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def compute_melspec(self, y):
        return self.amp_db_tran(self.mel_trans(y)).float()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, x, targets=None, do_mixup=False):
        # (batch_size, len_audio)
        with autocast(enabled=False):
            with torch.no_grad():
                x = self.compute_melspec(x)
                x = mono_to_color(x).transpose(1, 3)  # (batch_size, 3, time_steps, mel_bins)
                x = x - CFG.mean
                x = x / CFG.std

                if self.training and do_mixup:
                    if np.random.rand() < 0.5:
                        x, targets = mixup(x, targets, 0.4)
                    else:
                        x, targets = cutmix(x, targets, 0.4)

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

        x = F.dropout(x, p=CFG.dropout, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=CFG.dropout, training=self.training)

        clipwise_output, logit = self.att_block(x)

        output_dict = {
            'clipwise_output': clipwise_output,  # (n_samples, n_class)
            'logit': logit,  # (n_samples, n_class)
        }

        loss = None
        if targets is not None:
            if do_mixup:
                loss = mixup_criterion(output_dict["logit"], targets)
            else:
                loss = loss_fn(output_dict["logit"], targets)
        output_dict["loss"] = loss
        return output_dict
