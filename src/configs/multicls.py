from pathlib import Path

import numpy as np
import torch


class CFG:
    exp_name = "weight1-alpha05-seed1"  # this goes to the save filename
    output_dir = "exp/multiclass/"

    audios_path = "./data/train_audio/*/*.ogg"
    split_audios_path = "./data/train_np/"
    train_metadata = "./data/train_metadata.csv"  # making sure we use the same split
    train_labels = "./data/train_soundscapes.csv"

    nb_workers = 8
    period = 30
    n_mels = 224  # 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    in_chans = 1

    seed = 1
    epochs = 23
    cutmix_and_mixup_epochs = 18
    fold = 0  # [0, 1, 2, 3, 4]
    dropout = 0.5
    N_FOLDS = 5
    LR = 2 * 1e-3
    ETA_MIN = 1e-6
    WEIGHT_DECAY = 1e-6
    mixup_alpha = 0.4
    scored_weight = 1
    train_bs = 16  # 32
    valid_bs = 16  # 64
    base_model_name = "tf_efficientnet_b3_ns"
    starting_weights = None  # "fold-0-b3-779.bin"
    load_up_to_layer = None  # 1
    apex = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mean = (
        torch.from_numpy(np.array([0.485, 0.456, 0.406])).float()[None, :, None, None].to(device)
    )  # RGB
    std = (
        torch.from_numpy(np.array([0.229, 0.224, 0.225])).float()[None, :, None, None].to(device)
    )  # RG

    pretrained = True
    num_classes = 152
    target_columns = "afrsil1 akekee akepa1 akiapo akikik amewig aniani apapan arcter \
                      barpet bcnher belkin1 bkbplo bknsti bkwpet blkfra blknod bongul \
                      brant brnboo brnnod brnowl brtcur bubsan buffle bulpet burpar buwtea \
                      cacgoo1 calqua cangoo canvas caster1 categr chbsan chemun chukar cintea \
                      comgal1 commyn compea comsan comwax coopet crehon dunlin elepai ercfra eurwig \
                      fragul gadwal gamqua glwgul gnwtea golphe grbher3 grefri gresca gryfra gwfgoo \
                      hawama hawcoo hawcre hawgoo hawhaw hawpet1 hoomer houfin houspa hudgod iiwi incter1 \
                      jabwar japqua kalphe kauama laugul layalb lcspet leasan leater1 lessca lesyel lobdow lotjae \
                      madpet magpet1 mallar3 masboo mauala maupar merlin mitpar moudov norcar norhar2 normoc norpin \
                      norsho nutman oahama omao osprey pagplo palila parjae pecsan peflov perfal pibgre pomjae puaioh \
                      reccar redava redjun redpha1 refboo rempar rettro ribgul rinduc rinphe rocpig rorpar rudtur ruff \
                      saffin sander semplo sheowl shtsan skylar snogoo sooshe sooter1 sopsku1 sora spodov sposan \
                      towsol wantat1 warwhe1 wesmea wessan wetshe whfibi whiter whttro wiltur yebcar yefcan zebdov".split()
    scored_birds = [
        "akiapo",
        "aniani",
        "apapan",
        "barpet",
        "crehon",
        "elepai",
        "ercfra",
        "hawama",
        "hawcre",
        "hawgoo",
        "hawhaw",
        "hawpet1",
        "houfin",
        "iiwi",
        "jabwar",
        "maupar",
        "omao",
        "puaioh",
        "skylar",
        "warwhe1",
        "yefcan",
    ]
