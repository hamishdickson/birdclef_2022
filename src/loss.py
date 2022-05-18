from typing import Dict

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
            + (1.0 - targets) * probas ** self.gamma * bce_loss
        )
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    """

    def __init__(
        self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = (
            self.anti_targets
        ) = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(
                        1 - self.xs_pos - self.xs_neg,
                        self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
                    )
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(
                    1 - self.xs_pos - self.xs_neg,
                    self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
                )
                self.loss *= self.asymmetric_w
        # _loss = -self.loss.sum() / x.size(0)
        _loss = -self.loss / x.size(0)
        _loss = _loss / y.size(1) * 1000
        return _loss


def mixup_criterion(
    preds: torch.Tensor, new_targets: Dict[str, torch.Tensor], criterion: nn.Module
) -> torch.Tensor:
    targets1, targets2, lam, weights1, weights2 = (
        new_targets["targets1"],
        new_targets["targets2"],
        new_targets["lambda"],
        new_targets["weights1"],
        new_targets["weights2"],
    )
    loss1 = lam * criterion(preds, targets1)
    loss2 = (1 - lam) * criterion(preds, targets2)
    if weights1 is not None and weights2 is not None:
        loss1 = (loss1 * weights1.unsqueeze(-1)).sum(0) / weights1.sum()
        loss2 = (loss2 * weights2.unsqueeze(-1)).sum(0) / weights2.sum()
    return loss1.mean() + loss2.mean()
