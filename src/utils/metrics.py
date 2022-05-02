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

    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        ranges = np.arange(0.1, 1, 0.1)
        scores = [
            metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > x, average="micro")
            for x in ranges
        ]
        best = np.argmax(scores)
        return {
            "f1_at_03": scores[2],
            "f1_at_05": scores[4],
            "f1_at_best": (ranges[best], scores[best]),
        }
