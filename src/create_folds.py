import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
# from  IPython.display import Audio
from pathlib import Path

from matplotlib import pyplot as plt

from tqdm import tqdm
import joblib, json

from  sklearn.model_selection  import StratifiedKFold

SR = 32_000
SEED = 666

DATA_ROOT = "input"
TRAIN_AUDIO_ROOT = "input/train_audio"
TRAIN_AUDIO_IMAGES_SAVE_ROOT = "input/audio_images" # Where to save the mels images
# TRAIN_AUDIO_IMAGES_SAVE_ROOT.mkdir(exist_ok=True, parents=True)


def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames)/sr
    return {"frames": frames, "sr": sr, "duration": duration}


def make_df(n_splits=5, seed=SEED, nrows=None):
    
    df = pd.read_csv(f"{DATA_ROOT}/train_metadata.csv", nrows=nrows)

    LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df["primary_label"].unique()))}
    
#     df = df.iloc[PART_INDEXES[PART_ID]: PART_INDEXES[PART_ID+1]]

    df["filepath"] = [f"{TRAIN_AUDIO_ROOT}/{filename}" for primary_label,filename in zip(df.primary_label, df.filename) ]

    print(df.sample(3))

    print(get_audio_info('input/train_audio/afrsil1/XC125458.ogg'))

    pool = joblib.Parallel(4)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(filepath) for filepath in df.filepath]

    df = pd.concat([df, pd.DataFrame(pool(tqdm(tasks)))], axis=1, sort=False)
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    splits = skf.split(np.arange(len(df)), y=df.primary_label.values)
    df["fold"] = -1

    for fold, (train_set, val_set) in enumerate(splits):
        
        df.loc[df.index[val_set], "fold"] = fold

    return LABEL_IDS, df


if __name__ == "__main__":
    LABEL_IDS, df = make_df(nrows=None)

    df.to_csv("input/rich_train_metadata.csv", index=True)
    with open("input/LABEL_IDS.json", "w") as f:
        json.dump(LABEL_IDS, f)

    print(df.shape)
    print(df.sample(3))

    print(df["fold"].value_counts())
    print(df["primary_label"].value_counts())

    df.to_csv("input/train_metadata_new.csv")