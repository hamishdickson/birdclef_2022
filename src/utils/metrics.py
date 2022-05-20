import numpy as np
from tqdm import tqdm

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
    def __init__(self, optimise_per_bird=False, ranges=np.arange(0.025, 0.325, 0.025)):
        self.reset()
        self._optimise_per_bird = optimise_per_bird
        self._ranges = ranges

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred, mask=None):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    def score(self, y_true, y_pred, scored_birds=None, optim=False):
        if optim:
            best_ths = {}
            best_scores = {}
            print("Optimising threshold per bird...")
            for k in tqdm(range(len(CFG.target_columns))):
                bird = CFG.target_columns[k]
                if scored_birds is not None and k not in scored_birds:
                    continue
                scores_k = [
                    cmp_competition_metric(y_true[:, k, None], y_pred[:, k, None] > x)
                    for x in self._ranges
                ]
                best_k = np.argmax(scores_k)
                best_ths[bird] = self._ranges[best_k]
                best_scores[bird] = scores_k[best_k]
            return {
                "best_ths_per_bird": best_ths,
                "best_score_per_bird": best_scores,
                "optimised_global_score": np.mean(list(best_scores.values())),
            }
        else:
            scores = [
                cmp_competition_metric(y_true, y_pred > x, scored_classes=scored_birds)
                for x in self._ranges
            ]
            best = np.argmax(scores)
            return {
                "f1_at_best": (self._ranges[best], scores[best]),
            }

    @property
    def avg(self):
        # scores = self.score(self.y_true, self.y_pred, optim=False)
        scores = {}
        scored_birds = [CFG.target_columns.index(x) for x in CFG.scored_birds]
        scores_masked = self.score(
            np.array(self.y_true),
            np.array(self.y_pred),
            scored_birds,
            self._optimise_per_bird,
        )
        scores.update({f"masked_{k}": v for k, v in scores_masked.items()})
        return scores
