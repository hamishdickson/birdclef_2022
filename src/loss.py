import torch
import torch.nn as nn


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
               (1. - probas) ** self.gamma * bce_loss + \
               (1. - targets) * probas ** self.gamma * bce_loss
        loss = loss.mean()
        return loss


def mixup_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = FocalLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def loss_fn(logits, targets):
    loss_fct = FocalLoss()
    loss = loss_fct(logits, targets)
    return loss
