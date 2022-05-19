import os
import random
import time

import numpy as np
import torch


class Timer:
    def __init__(self, name, print_freq=None):
        self.name = name
        self.print_freq = print_freq
        self.dur = 0
        self.count = 0

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args, **kwargs):
        self.dur = time.time() - self._start
        self.count += 1
        if self.print_freq is not None and not (self.count + 1) % self.print_freq:
            print(f"{self.name} timer: {1000 * self.dur / self.count:.5f}ms per call")


LOADTIMER = Timer("load")
AUGTIMER = Timer("aug")
SPECTIMER = Timer("spec")
ALBUTIMER = Timer("albu")


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha, weights=None, strong_targets=None):

    indices = torch.randperm(data.size(0))
    shuffled_targets = targets[indices]

    if weights is not None:
        shuffled_weights = weights[indices]
    else:
        shuffled_weights = None

    if strong_targets is not None:
        shuffled_strong_targets = strong_targets[indices]
    else:
        shuffled_strong_targets = None

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = {
        "targets1": targets,
        "targets2": shuffled_targets,
        "lambda": lam,
        "weights1": weights,
        "weights2": shuffled_weights,
        "strong_targets1": strong_targets,
        "strong_targets2": shuffled_strong_targets,
    }
    return data, new_targets


def mixup(data, targets, alpha, weights=None, strong_targets=None):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    if weights is not None:
        shuffled_weights = weights[indices]
    else:
        shuffled_weights = None

    if strong_targets is not None:
        shuffled_strong_targets = strong_targets[indices]
    else:
        shuffled_strong_targets = None

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    new_targets = {
        "targets1": targets,
        "targets2": shuffled_targets,
        "lambda": lam,
        "weights1": weights,
        "weights2": shuffled_weights,
        "strong_targets1": strong_targets,
        "strong_targets2": shuffled_strong_targets,
    }
    return new_data, new_targets
