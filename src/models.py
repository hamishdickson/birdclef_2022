import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import transformers
import numpy as np

from torch.cuda.amp import autocast

class BaselineModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super(BaselineModel, self).__init__()
        self.base_model = timm.create_model(
            config.model_name, pretrained=pretrained, in_chans=3#, img_size=(128, 512)
        )
        self.base_model.fc = nn.Linear(2048, 21)


    def forward(self, in_x):
        with autocast():
            x = self.base_model(in_x)
            return x