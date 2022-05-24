import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from nnAudio import Spectrogram
from nnAudio.features import CQT1992v2
from torch.cuda.amp import autocast
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchlibrosa.augmentation import SpecAugmentation

from . import loss as loss_module
from .layers import AttBlockV2, init_bn, init_layer
from .utils.general import cutmix, mixup
from .utils.transforms import mono_to_color


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """

    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)
    return output


def build_loss_fn(loss_name, **loss_kwargs):
    return getattr(loss_module, loss_name)(**loss_kwargs)


class TimmReshape(nn.Module):
    def __init__(self, base_model_name: str, cfg, pretrained=False, num_classes=24):
        super().__init__()
        assert cfg.period // 5, "Input duration must be divisible by 5"
        self._divisor = cfg.period // 5
        self.cfg = cfg
        loss_name = cfg.loss_name
        if loss_name == "FocalLoss":
            self._loss_fn = build_loss_fn(loss_name, **cfg.focal_kwargs)
        elif loss_name == "AsymmetricLoss":
            self._loss_fn = build_loss_fn(loss_name, **cfg.asym_kwargs)

        self.mel_trans = MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
        )
        self.amp_db_tran = AmplitudeToDB()

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64 // 2, time_stripes_num=2, freq_drop_width=8 // 2, freq_stripes_num=2
        )

        self.bn0 = nn.BatchNorm2d(self.cfg.n_mels)
        encoder_kwargs = {
            "model_name": base_model_name,
            "in_chans": self.cfg.in_chans,
            "pretrained": pretrained,
        }
        if "resnest" not in base_model_name:
            encoder_kwargs.update({"drop_path_rate": self.cfg.drop_path})
        self.encoder = timm.create_model(**encoder_kwargs)

        in_features = self.encoder.num_features
        self.encoder.reset_classifier(0, "")
        self.fc = nn.Linear(in_features, num_classes)

    def compute_melspec(self, y):
        return self.amp_db_tran(self.mel_trans(y)).float()

    def forward(self, waveform, targets=None, do_mixup=False, weights=None):
        # (batch_size, len_audio)
        with autocast(enabled=False):
            with torch.no_grad():
                x = self.compute_melspec(waveform)
                x = x.unsqueeze(1)
                # melspec + d1 + d2
                delta1 = AF.compute_deltas(x)
                delta2 = AF.compute_deltas(delta1)
                x = torch.cat([x, delta1, delta2], 1)
                _min, _max = x.amin(dim=(2, 3), keepdim=True), x.amax(dim=(2, 3), keepdim=True)
                x = x.transpose(2, 3)
                x = (x - _min) / (_max - _min)
                if self.training and do_mixup:
                    if np.random.rand() < 0.5:
                        x, targets = mixup(x, targets, self.cfg.mixup_alpha, weights=weights)
                    else:
                        x, targets = cutmix(x, targets, self.cfg.mixup_alpha, weights=weights)

        pad_h = int(np.ceil(x.shape[2] / self._divisor)) * self._divisor
        pad = (0, 0, 0, max(pad_h - x.shape[2], 0))
        x = F.pad(x, pad, "constant", 0)
        bs, c, h, w = x.shape
        x = (
            x.reshape(bs, c, self._divisor, h // self._divisor, w)
            .permute(0, 2, 1, 3, 4)
            .flatten(0, 1)
        )
        x = x.transpose(1, 3)  # (batch_size, mel_bins, time_steps, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (batch_size, 3, time_steps, mel_bins)
        if self.training:
            if np.random.random() < 0.25:
                x = self.spec_augmenter(x)

        x = self.encoder(x)
        x = x.reshape(bs, self._divisor, x.shape[1], x.shape[2], x.shape[3])
        logit = self.fc(x.mean((1, 3, 4)))
        clipwise_output = torch.sigmoid(logit)
        segmentwise_output = torch.sigmoid(self.fc(x.mean((3, 4))))
        output_dict = {
            "segmentwise_output": segmentwise_output,
            "clipwise_output": clipwise_output,  # (n_samples, n_class)
            "logit": logit,  # (n_samples, n_class)
        }

        if self.training:
            if do_mixup:
                loss = loss_module.mixup_criterion(output_dict["logit"], targets, self._loss_fn)
            else:
                loss = self._loss_fn(output_dict["logit"], targets)
                if weights is not None:
                    loss = (loss * weights.unsqueeze(-1)).sum(0) / weights.sum()
                loss = loss.mean()
        else:
            if targets is not None:
                loss = self._loss_fn(output_dict["logit"], targets)
                loss = loss.mean()
            else:
                loss = None
        output_dict["loss"] = loss
        return output_dict


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, cfg, pretrained=False, num_classes=24):
        super().__init__()
        self.cfg = cfg
        loss_name = cfg.loss_name
        mel_type = cfg.melspec_type
        assert mel_type in ["delta", "color", "power", "mono"]

        if loss_name == "FocalLoss":
            self._loss_fn = build_loss_fn(loss_name, **cfg.focal_kwargs)
        elif loss_name == "AsymmetricLoss":
            self._loss_fn = build_loss_fn(loss_name, **cfg.asym_kwargs)

        self.register_buffer("mean", cfg.mean, persistent=False)
        self.register_buffer("std", cfg.std, persistent=False)

        if mel_type == "power":
            self.mel_trans_power = Spectrogram.MelSpectrogram(
                sr=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                fmin=self.cfg.fmin,
                fmax=self.cfg.fmax,
                n_mels=self.cfg.n_mels,
                power=2.0,
                trainable_mel=True,
                trainable_STFT=True,
                verbose=False,
            )
            self.mel_trans_energy = Spectrogram.MelSpectrogram(
                sr=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                fmin=self.cfg.fmin,
                fmax=self.cfg.fmax,
                n_mels=self.cfg.n_mels,
                power=1.0,
                trainable_mel=True,
                trainable_STFT=True,
                verbose=False,
            )
        elif mel_type == "powercqt":
            self.cqt = CQT1992v2(
                sr=self.cfg.sample_rate,
                hop_length=self.cfg.hop_length,
                fmin=self.cfg.fmin,
                fmax=self.cfg.fmax,
                bins_per_octave=24,
                verbose=False,
                trainable=True,
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
                trainable_STFT=True,
            )
            self.mel_trans = MelSpectrogram(
                sr=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                fmin=self.cfg.fmin,
                fmax=self.cfg.fmax,
                n_mels=self.cfg.n_mels,
            )
            self.pcen_trans = PCENTransform(
                eps=1e-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True
            )
        else:
            self.mel_trans = MelSpectrogram(
                sample_rate=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                f_min=self.cfg.fmin,
                f_max=self.cfg.fmax,
                n_mels=self.cfg.n_mels,
            )
        self.amp_db_tran = AmplitudeToDB()

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64 // 2, time_stripes_num=2, freq_drop_width=8 // 2, freq_stripes_num=2
        )

        self.bn0 = nn.BatchNorm2d(self.cfg.n_mels)

        encoder_kwargs = {
            "model_name": base_model_name,
            "in_chans": self.cfg.in_chans,
            "pretrained": pretrained,
        }
        if "resnest" not in base_model_name:
            encoder_kwargs.update({"drop_path_rate": self.cfg.drop_path})
        self.encoder = timm.create_model(**encoder_kwargs)

        in_features = self.encoder.num_features
        self.encoder.reset_classifier(0, "")

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def compute_melspec(self, y):
        return self.amp_db_tran(self.mel_trans(y)).float()

    def compute_melspec_multi_channel(self, y):
        return torch.stack(
            (
                self.amp_db_tran(self.mel_trans_power(y)).float(),
                self.amp_db_tran(self.mel_trans_energy(y)).float(),
                self.amp_db_tran(self.mel_trans_power(y)).float(),
            ),
            1,
        )

    def compute_melspec_cqt(self, y):
        return torch.stack(
            (
                self.amp_db_tran(self.mel_trans_power(y)).float(),
                self.pcen_trans(self.mel_trans(y)).float(),
                self.cqt(y).float()[:, : self.cfg.n_mels, :],
            ),
            1,
        )

    def preprocess_audio(self, waveform: torch.Tensor, type="color"):
        if "power" in type:
            if type == "power":
                x = self.compute_melspec_multi_channel(waveform)
            elif type == "powercqt":
                x = self.compute_melspec_cqt(waveform)
            _min, _max = x.amin(dim=(1, 2, 3), keepdim=True), x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - _min) / (_max - _min)
            x = (x - self.mean) / self.std
        else:
            x = self.compute_melspec(waveform)
            if type == "color":
                x = mono_to_color(x)
                x = (x - self.mean) / self.std
            elif type == "delta":
                delta1 = AF.compute_deltas(x)
                delta2 = AF.compute_deltas(delta1)
                x = torch.stack([x, delta1, delta2], 1)
                _min, _max = x.amin(dim=(2, 3), keepdim=True), x.amax(dim=(2, 3), keepdim=True)
                x = (x - _min) / (_max - _min)
            elif type == "mono":
                x = x.unsqueeze(1)
                _min, _max = x.amin(dim=(1, 2, 3), keepdim=True), x.amax(
                    dim=(1, 2, 3), keepdim=True
                )
                x = (x - _min) / (_max - _min)

        x = x.transpose(2, 3)
        return x

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, waveform, targets=None, do_mixup=False, weights=None):
        # (batch_size, len_audio)
        with autocast(enabled=False):
            with torch.no_grad():
                # (batch_size, in_chans, time_steps, mel_bins)
                x = self.preprocess_audio(waveform, self.cfg.melspec_type)
                frames_num = x.shape[2]
                if self.training and do_mixup:
                    if np.random.rand() < 0.5:
                        x, targets = mixup(x, targets, self.cfg.mixup_alpha, weights=weights)
                    else:
                        x, targets = cutmix(x, targets, self.cfg.mixup_alpha, weights=weights)

        x = x.transpose(1, 3)  # (batch_size, mel_bins, time_steps, in_chans)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (batch_size, in_chans, time_steps, mel_bins)

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

        # Extract segmentwise and framewise
        clipwise_logit, _ = self.att_block(x)
        clipwise_output = torch.sigmoid(clipwise_logit)
        output_dict = {
            "clipwise_output": clipwise_output,  # (n_samples, n_class)
            "logit": clipwise_logit,
        }

        if self.training:
            if do_mixup:
                loss = loss_module.mixup_criterion(output_dict["logit"], targets, self._loss_fn)
            else:
                loss = self._loss_fn(output_dict["logit"], targets)
                if weights is not None:
                    loss = (loss * weights.unsqueeze(-1)).sum(0) / weights.sum()
                loss = loss.mean()
        else:
            if targets is not None:
                loss = self._loss_fn(output_dict["logit"], targets)
                loss = loss.mean()
            else:
                loss = None
        output_dict["loss"] = loss
        return output_dict


def pcen(x, eps=1e-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
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
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta**r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta**r)
    return pcen_


class PCENTransform(nn.Module):
    def __init__(self, eps=1e-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):
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
        x = x.transpose(2, 1)
        if self.trainable:
            x = pcen(
                x,
                self.eps,
                torch.exp(self.log_s),
                torch.exp(self.log_alpha),
                torch.exp(self.log_delta),
                torch.exp(self.log_r),
                self.training and self.trainable,
            )
        else:
            x = pcen(
                x,
                self.eps,
                self.s,
                self.alpha,
                self.delta,
                self.r,
                self.training and self.trainable,
            )
        x = x.transpose(2, 1)
        return x
