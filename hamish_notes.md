# bird comp 2022

https://www.kaggle.com/c/birdclef-2022

compute:
threadripper 2970WX
128 GB ram
rtx 3090 FE

## subs

| CV | LB | Name | Notes |
|---- | -----| --- | --- |
| 0.6858006854896994 | 0.60 | exp2 | old CV strategy, 5fold, wrong thresh (0.025) |
| 0.6858006854896994 | ??? | exp3 | exp2 but with correct thresh (0.049726561999999995) not changed threshlong |


## metric

https://www.kaggle.com/c/birdclef-2022/discussion/309408

```python
import numpy as np
from typing import List

def f1_score(true_pos, false_neg, false_pos):
    true_pos, false_neg, false_pos = float(true_pos), float(false_neg), float(false_pos)
    if true_pos == 0 and (false_neg + false_pos) == 0:
        return 1.0
    else:
        return 2 * true_pos / (2 * true_pos + false_neg + false_pos)

def macro_f1_similarity(
    y_true_all_episodes: List[List[int]], y_pred_all_episodes: List[List[int]], unique_bird_ids: List[int]
) -> float:
    assert len(y_true_all_episodes) == len(y_pred_all_episodes), "different number of episodes for true and pred"
    n_episodes = len(y_true_all_episodes)

    stats_dict = {k:{"tp":0, "fn":0, "fp":0} for k in unique_bird_ids}

    for episode_idx in range(n_episodes):
        for true_elem in y_true_all_episodes[episode_idx]:
            if true_elem in unique_bird_ids:
                if true_elem in y_pred_all_episodes[episode_idx]:
                    stats_dict[true_elem]["tp"] += 1
                else:
                    stats_dict[true_elem]["fn"] += 1

        for pred_el in y_pred_all_episodes[episode_idx]:
            if pred_el in unique_bird_ids:
                if pred_el not in y_true_all_episodes[episode_idx]:
                    stats_dict[pred_el]["fp"] += 1


    f1_similarity = np.mean([f1_score(
        true_pos=item["tp"], 
        false_neg=item["fn"], 
        false_pos=item["fp"]
    ) for item in stats_dict.values()])


    return f1_similarity
```

## notes

## 220227

have to retrain, got the wrong number of classes (should have been 152)
exp2 + exp3
[0.692622163355923, 0.6844865818948619, 0.6865796493069221, 0.6850770217436883, 0.6802380111471019] = 0.6858006854896994
thresh
[0.03769531, 0.05009766, 0.05566406, 0.05126953, 0.05390625] = 0.049726561999999995

5-fold with below config, 2021 cv: [0.6249072928642098, 0.6072954103751815, 0.6281849276798772, 0.6140257068034846, 0.6164860485567556] = 0.6181798772559017

```python
CONFIG = {
    "seed": 42,
    "n_fold": 5,
    "epochs": 10,
    "batch_size": 32,
    "n_accumulate": 1,
    "n_workers": 16,
    "model_save_name": "baseline",
    "model_name": "resnet34",  # <---- just for you Eduardo
    "lr": 2e-4,
    "backbone_lr": 1e-4,
    "weight_decay": 0,
    "warmup": 0,
    "sample": False,
    "max_grad": 0,
    "num_class": 397,
    "train_period": 30.0,
    "infer_period": 30.0,
    "p": 0.5,
    "n_mels": 128,
    "mixup_p": 0.5,
    "mixup_alpha": 0.8,
}
```

## 220226

start implementing this as strong baseline https://github.com/tattaka/birdclef-2021

some thoughts:
- they prebuild the mel specs before training, I guess this is fine but perhaps it makes more sense to do on the fly with something like nnAudio (GPU)

## 220225

More hacking

## 220224

start trying to hack out a solution

> only predictions for the 21 species listed in scored_birds.json are scored for this competition, predictions for other species will be dropped from your submission.

Aux labels it is then