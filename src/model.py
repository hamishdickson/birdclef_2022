import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchaudio.transforms import AmplitudeToDB #, MelSpectrogram
from torchlibrosa.augmentation import SpecAugmentation

from nnAudio.Spectrogram import MelSpectrogram, CQT1992v2

from layers import AttBlockV2, init_bn, init_layer
from loss import loss_fn, mixup_criterion
from utils.general import cutmix, mixup
from utils.transforms import mono_to_color, multi_norm

def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
    frames = x.split(1, -2)
    m_frames = []
    last_state = None
    for frame in frames:
        if last_state is None:
            last_state = s * frame
            m_frames.append(last_state)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add_(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    if training:
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_


class PCENTransform(nn.Module):

    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):
        super().__init__()
        if trainable:
            self.log_s = nn.Parameter(torch.log(torch.Tensor([s])))
            self.log_alpha = nn.Parameter(torch.log(torch.Tensor([alpha])))
            self.log_delta = nn.Parameter(torch.log(torch.Tensor([delta])))
            self.log_r = nn.Parameter(torch.log(torch.Tensor([r])))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable

    def forward(self, x):
        x = x.transpose(2,1)
        if self.trainable:
            x = pcen(x, self.eps, torch.exp(self.log_s), torch.exp(self.log_alpha), torch.exp(self.log_delta), torch.exp(self.log_r), self.training and self.trainable)
        else:
            x = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable)
        x = x.transpose(2,1)
        return x


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, cfg, pretrained=False, num_classes=24):
        super().__init__()
        self.cfg = cfg

        self.cqt = CQT1992v2(
            sr=self.cfg.sample_rate,
            # n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            bins_per_octave=24,
            # n_mels=self.cfg.n_mels,
            # power=1.5,
            trainable=True,
            # trainable_STFT=True
        )

        self.mel_trans_power = MelSpectrogram(
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
            power=2.0,
            trainable_mel=True,
            trainable_STFT=True
        )

        self.mel_trans = MelSpectrogram(
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
            # power=1.0,
            # trainable_mel=True,
            # trainable_STFT=True
        )


        self.pcen_trans = PCENTransform(eps=1E-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True)

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
        return (self.amp_db_tran(self.mel_trans_power(y)).float(), self.pcen_trans(self.mel_trans(y)).float(), self.cqt(y).float()[:,:self.cfg.n_mels,:])

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
                    if np.random.rand() < self.cfg.mixup_perc:
                        x, targets = mixup(x, targets, self.cfg.mixup_alpha, weights=weights)
                    else:
                        x, targets = cutmix(x, targets, self.cfg.mixup_alpha, weights=weights)

        x = x.transpose(1, 3)  # (batch_size, mel_bins, time_steps, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (batch_size, 3, time_steps, mel_bins)

        if self.training and do_mixup:
            if np.random.random() < self.cfg.spec_augmenter:
                x = self.spec_augmenter(x)

        x = self.encoder(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)  # <- actually agg in the timestep axis

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        if do_mixup:
            x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
    
        if do_mixup:
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

