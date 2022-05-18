import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.cuda.amp import autocast
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchlibrosa.augmentation import SpecAugmentation

from .layers import AttBlockV2, init_bn, init_layer
from .loss import loss_fn, mixup_criterion
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


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, cfg, pretrained=False, num_classes=24):
        super().__init__()
        self.cfg = cfg

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

        self.encoder = timm.create_model(
            base_model_name, in_chans=self.cfg.in_chans, pretrained=pretrained
        )

        in_features = self.encoder.num_features
        self.encoder.reset_classifier(0, "")

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def compute_melspec(self, y):
        return self.amp_db_tran(self.mel_trans(y)).float()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, waveform, targets=None, do_mixup=False, weights=None):
        # (batch_size, len_audio)
        with autocast(enabled=False):
            with torch.no_grad():
                x = self.compute_melspec(waveform)
                x = x.unsqueeze(1)
                # melspec
                # _min, _max = x.amin(dim=(1, 2, 3), keepdim=True), x.amax(
                #     dim=(1, 2, 3), keepdim=True
                # )
                # melspec + d1 + d2
                delta1 = AF.compute_deltas(x)
                delta2 = AF.compute_deltas(delta1)
                x = torch.cat([x, delta1, delta2], 1)
                _min, _max = x.amin(dim=(2, 3), keepdim=True), x.amax(dim=(2, 3), keepdim=True)

                frames_num = x.shape[2]
                x = x.transpose(2, 3)
                x = (x - _min) / (_max - _min)

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

        # Extract segmentwise and framewise
        clipwise_output, logit, segmentwise_logit = self.att_block(x)
        segmentwise_logit = segmentwise_logit.transpose(1, 2)
        interpolate_ratio = frames_num // segmentwise_logit.size(1)
        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_output = torch.sigmoid(framewise_logit)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {
            "clipwise_output": clipwise_output,  # (n_samples, n_class)
            "framewise_output": framewise_output,
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
