import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import transformers
import numpy as np

from torch.cuda.amp import autocast

from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    """
    Simple class for flattening layer.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V


class AttHead(nn.Module):
    def __init__(
        self, in_chans, p=0.5, num_class=397, train_period=15.0, infer_period=5.0
    ):
        super().__init__()
        self.train_period = train_period
        self.infer_period = infer_period
        self.pooling = GeMFreq()

        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, 512),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)  # (bs, time, ch)

        feat = self.dense_layers(feat).permute(0, 2, 1)  # (bs, 512, time)
        time_att = torch.tanh(self.attention(feat))
        assert self.train_period >= self.infer_period
        if self.training or self.train_period == self.infer_period:

            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # sum((bs, 24, time), -1) -> (bs, 24)
            logits = torch.sum(
                self.fix_scale(feat) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )
        else:
            feat_time = feat.size(-1)
            start = (
                feat_time / 2 - feat_time * (self.infer_period / self.train_period) / 2
            )
            end = start + feat_time * (self.infer_period / self.train_period)
            start = int(start)
            end = int(end)
            feat = feat[:, :, start:end]
            att = torch.softmax(time_att[:, :, start:end], dim=-1)
            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * att,
                dim=-1,
            )
            logits = torch.sum(
                self.fix_scale(feat) * att,
                dim=-1,
            )
            time_att = time_att[:, :, start:end]
        return (
            logits,
            clipwise_pred,
            self.fix_scale(feat).permute(0, 2, 1),
            time_att.permute(0, 2, 1),
        )


class AttModel(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        p=0.5,
        n_mels=224,
        num_class=152,
        train_period=15.0,
        infer_period=5.0,
        in_chans=1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.logmelspec_extractor = nn.Sequential(
            MelSpectrogram(
                32000,
                n_mels=n_mels,
                f_min=20,
                n_fft=2048,
                hop_length=512,
                normalized=True,
            ),
            AmplitudeToDB(top_db=80.0),
            NormalizeMelSpec(),
        )
        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=True, in_chans=in_chans
        )
        encoder_channels = self.backbone.feature_info.channels()
        dense_input = encoder_channels[-1]
        self.head = AttHead(
            dense_input,
            p=p,
            num_class=num_class,
            train_period=train_period,
            infer_period=infer_period,
        )

    def forward(self, input):
        feats = self.backbone(input)
        return self.head(feats[-1])


class BaselineModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super(BaselineModel, self).__init__()
        self.att_model = AttModel(
            config["model_name"],
            p=config["p"],
            n_mels=config["n_mels"],
            num_class=config["num_class"],
            train_period=config["train_period"],
            infer_period=config["infer_period"],
        )

    def forward(self, image):
        with autocast():
            outputs = {}
            (
                outputs["logits"],
                outputs["output_clip"],
                outputs["output_frame"],
                outputs["output_attention"],
            ) = self.att_model(image)
            return outputs
