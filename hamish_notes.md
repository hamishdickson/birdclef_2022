# bird comp 2022

https://www.kaggle.com/c/birdclef-2022

compute:
threadripper 2970WX
128 GB ram
rtx 3090 FE

## subs

| CV | LB | Name | Notes |
|---- | -----| --- | --- |
| 0.6858006854896994 | 0.60 | hwd2 | old CV strategy, 5fold, wrong thresh (0.025) |
| 0.6858006854896994 | 0.58 | hwd3 | exp2 but with correct thresh (0.049726561999999995) not changed threshlong |
| 0.62 | 0.61 | hwd4 | 21 + aux classes, still working on CV score, subing to confirm everything works |

## notes

## 220318

try to train against just the 21+aux classes

hwd4

## 220227

have to retrain, got the wrong number of classes (should have been 152)
hwd2 + hwd3
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