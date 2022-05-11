import numpy as np

from ..configs.multicls import CFG


def cmp_competition_metric(y_true, y_pred, scored_classes=None):
    """
    From eval description:
    "we technically run a weighted classification accuracy with the weights set such that
    all of the species are assigned the same total weight and the true negatives and
    true positives for each species have the same weight"
    """
    y_true = np.array(y_true) > 0.5  # make sure is hard label, shape: samples, nb_classes
    y_pred = np.array(y_pred)

    assert y_pred.shape == y_true.shape, f"pred: {y_pred.shape}, true: {y_true.shape}"
    matches = (y_true == y_pred).astype(int)

    if not scored_classes:
        scored_classes = range(y_pred.shape[1])

    classes_score = []
    for i in scored_classes:
        match_i = matches[:, i]
        gt_i = y_true[:, i]

        if gt_i.astype(int).mean() == 0:
            # print(f"Skipping class {i}")
            continue

        pos_score = match_i[gt_i].mean()
        neg_score = match_i[~gt_i].mean()
        classes_score.append((pos_score + neg_score) / 2)
    return np.mean(classes_score)


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

    def update(self, y_true, y_pred, mask=None):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @staticmethod
    def score(y_true, y_pred, ranges=np.arange(0.025, 0.325, 0.025), scored_birds=None):
        scores = [
            cmp_competition_metric(
                np.array(y_true), np.array(y_pred) > x, scored_classes=scored_birds
            )
            for x in ranges
        ]
        best = np.argmax(scores)
        return {
            "f1_at_best": (ranges[best], scores[best]),
        }

    @property
    def avg(self):
        scores = self.score(self.y_true, self.y_pred)
        scores_masked = self.score(
            self.y_true,
            self.y_pred,
            scored_birds=[CFG.target_columns.index(x) for x in CFG.scored_birds],
        )
        scores.update({f"masked_{k}": v for k, v in scores_masked.items()})
        return scores
