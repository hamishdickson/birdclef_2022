import numpy as np
import torch
import torch.nn as nn


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas**self.gamma * bce_loss
        )
        return loss


def mixup_criterion(logits, new_targets, framewise_weight=1):
    clipwise_logit, framewise_logit = logits["logit"], logits["framewise_logit"]

    targets1, targets2, lam, weights1, weights2, strong_targets1, strong_targets2 = (
        new_targets["targets1"],
        new_targets["targets2"],
        new_targets["lambda"],
        new_targets["weights1"],
        new_targets["weights2"],
        new_targets["strong_targets1"],
        new_targets["strong_targets2"],
    )

    criterion = FocalLoss(alpha=0.5)
    loss1 = lam * criterion(clipwise_logit, targets1)
    loss2 = (1 - lam) * criterion(clipwise_logit, targets2)
    # if weights1 is not None and weights2 is not None:
    #     loss1 = (loss1 * weights1.unsqueeze(-1)).sum(0) / weights1.sum()
    #     loss2 = (loss2 * weights2.unsqueeze(-1)).sum(0) / weights2.sum()
    clipwise_loss = loss1.mean() + loss2.mean()

    frame_loss1 = lam * criterion(framewise_logit, strong_targets1)
    frame_loss2 = (1 - lam) * criterion(framewise_logit, strong_targets2)
    framewise_loss = frame_loss1.mean() + frame_loss2.mean()
    if np.random.random() < 0.025:
        print(clipwise_loss, framewise_loss)
    return clipwise_loss + framewise_weight * framewise_loss


def loss_fn(logits, targets, weights=None, strong_targets=None, framewise_weight=1):
    clipwise_logit, framewise_logit = logits["logit"], logits["framewise_logit"]
    loss_fct = FocalLoss(alpha=0.5)

    clipwise_loss = loss_fct(clipwise_logit, targets)
    # if weights is not None:
    #     clipwise_loss  = (clipwise_loss * weights.unsqueeze(-1)).sum(0) / weights.sum()
    clipwise_loss = clipwise_loss.mean()
    loss = clipwise_loss

    if strong_targets is not None:
        framewise_loss = loss_fct(framewise_logit, strong_targets)
        framewise_loss = framewise_loss.mean()
        loss = loss + framewise_weight * framewise_loss
    return loss
