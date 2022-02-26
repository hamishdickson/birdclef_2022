import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import scipy as sp
from sklearn.metrics import f1_score


def set_seeds(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Mixup(object):
    def __init__(self, p=0.5, alpha=5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0


def row_wise_f1_score_micro(y_true, y_pred, threshold=0.5):
    def event_thresholder(x, threshold):
        return x > threshold

    return f1_score(
        y_true=y_true, y_pred=event_thresholder(y_pred, threshold), average="samples"
    )


class ThresholdOptimizer:
    def __init__(self, loss_fn):
        self.coef_ = {}
        self.loss_fn = loss_fn
        self.coef_["x"] = [0.5]

    def _loss(self, coef, X, y):
        ll = self.loss_fn(y, X, coef)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._loss, X=X, y=y)
        initial_coef = [0.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )

    def coefficients(self):
        return self.coef_["x"]

    def calc_score(self, X, y, coef):
        return self.loss_fn(y, X, coef)
