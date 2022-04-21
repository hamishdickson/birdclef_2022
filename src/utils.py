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


import numpy as np

from sklearn.metrics import jaccard_score


def comp_metric(target, pred, threshold=0.5):
    pred = pred >= threshold
    scores = []
    scores += [jaccard_score(target[:, i], pred[:, i], pos_label=1) for i in range(1, pred.shape[1])]
    scores += [jaccard_score(target[:, i], pred[:, i], pos_label=0) for i in range(1, pred.shape[1])]
    return np.mean(scores)

def comp_metric_old(y_true, y_pred, threshold=0.5):
    """
    Submissions are evaluated on a metric that is most similar to the macro F1 score. Given the amount of audio 
    data used in this competition it wasn't feasible to label every single species found in every soundscape. 
    Instead only a subset of species are actually scored for any given audio file. After dropping all of the 
    un-scored rows we technically run a weighted classification accuracy with the weights set such that all of 
    the species are assigned the same total weight and the true negatives and true positives for each species 
    have the same weight. The extra complexity exists purely to allow us to have a great deal of control over 
    which birds are scored for a given soundscape. For offline cross validation purposes, the macro F1 is the 
    closest analogue to the actual metric.
    """

    # don't think this is now needed?
    def event_thresholder(x, threshold):
        return x > threshold

    return f1_score(
        y_true=y_true[:,1:], y_pred=event_thresholder(y_pred[:,1:], threshold), average="samples"
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
