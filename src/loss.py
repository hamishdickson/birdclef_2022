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


def mixup_criterion(preds, new_targets):
    targets1, targets2, lam, weights1, weights2 = (
        new_targets["targets1"],
        new_targets["targets2"],
        new_targets["lambda"],
        new_targets["weights1"],
        new_targets["weights2"],
    )

    criterion = FocalLoss()
    loss1 = lam * criterion(preds, targets1)
    loss2 = (1 - lam) * criterion(preds, targets2)
    if weights1 is not None and weights2 is not None:
        loss1 = (loss1 * weights1.unsqueeze(-1)).sum(0) / weights1.sum()
        loss2 = (loss2 * weights2.unsqueeze(-1)).sum(0) / weights2.sum()
    return loss1.mean() + loss2.mean()


def loss_fn(logits, targets, weights=None):
    loss_fct = FocalLoss()
    loss = loss_fct(logits, targets)
    if weights is not None:
        loss = (loss * weights.unsqueeze(-1)).sum(0) / weights.sum()
    return loss.mean()
