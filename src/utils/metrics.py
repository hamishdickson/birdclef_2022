import numpy as np
from sklearn import metrics


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


class MetricMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_true_masked = []
        self.y_pred_masked = []

    def update(self, y_true, y_pred, mask=None):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())
        if mask is not None:
            self.y_true_masked.extend(y_true[mask].cpu().detach().numpy().tolist())
            self.y_pred_masked.extend(y_pred[mask].cpu().detach().numpy().tolist())

    @staticmethod
    def score(y_true, y_pred, ranges=np.arange(0.1, 1, 0.1)):
        scores = [
            metrics.f1_score(np.array(y_true), np.array(y_pred) > x, average="micro")
            for x in ranges
        ]
        best = np.argmax(scores)
        return {
            "f1_at_03": scores[2],
            "f1_at_05": scores[4],
            "f1_at_best": (ranges[best], scores[best]),
        }

    @property
    def avg(self):
        scores = self.score(self.y_true, self.y_pred)
        if len(self.y_true_masked):
            scores_masked = self.score(self.y_true_masked, self.y_pred_masked)
            scores.update({f"masked_{k}": v for k, v in scores_masked.items()})
        return scores
