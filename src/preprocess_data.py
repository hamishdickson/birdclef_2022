from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from tqdm import tqdm

from utils.transforms import load_audio


def process_file(path: Path, save_root: Path, clip_duration=5, sr=32000):
    save_dir = save_root / str(path).split("/")[-2]
    save_dir.mkdir(exist_ok=True, parents=True)

    audio = load_audio(str(path), target_sr=sr)
    nb_chunks = int(np.ceil(len(audio) / sr / clip_duration))
    for i in range(nb_chunks):
        start = i * clip_duration
        end = (i + 1) * clip_duration
        y = audio[start * sr : end * sr]
        sf.write(
            save_dir / str(path).split("/")[-1].replace(".ogg", f"_{str(end)}.ogg"),
            y,
            samplerate=sr,
        )


def main(args):
    input_dir = Path(args.input)
    paths = sorted(list(input_dir.glob("**/*.ogg")))

    pool = joblib.Parallel(args.nb_workers)
    mapper = joblib.delayed(
        lambda x: process_file(
            x, save_root=Path(args.output), clip_duration=args.duration, sr=args.sample_rate
        )
    )
    tasks = [mapper(path) for path in paths]

    pool(tqdm(tasks))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input root dir",
        default="/media/nvme/Datasets/bird/2022/train_audio",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output root dir",
        default="/media/nvme/Datasets/bird/2022/train_np",
    )
    parser.add_argument("-sr", "--sample_rate", type=int, default=32000)
    parser.add_argument("-d", "--duration", type=int, default=5)
    parser.add_argument("--nb_workers", type=int, default=12)
    args = parser.parse_args()
    main(args)
