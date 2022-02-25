# bird comp 2022

https://www.kaggle.com/c/birdclef-2022

compute:
threadripper 2970WX
128 GB ram
rtx 3090 FE

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

## 220225

More hacking

## 220224

start trying to hack out a solution

> only predictions for the 21 species listed in scored_birds.json are scored for this competition, predictions for other species will be dropped from your submission.

Aux labels it is then