import torch
import numpy as np


class CFG:
    EXP_ID = 'EX005'

    audios_path = "/media/nvme/Datasets/bird/2022/train_audio/*/*.ogg"
    train_metadata = "/media/nvme/Datasets/bird/2022/train_metadata.csv"
    train_labels = "/media/nvme/Datasets/bird/2022/audio_images5/train_soundscapes.csv"

    seed = 71
    epochs = 23
    cutmix_and_mixup_epochs = 18
    fold = 0  # [0, 1, 2, 3, 4]
    dropout = 0.1
    N_FOLDS = 5
    LR = 2 * 1e-3
    ETA_MIN = 1e-6
    WEIGHT_DECAY = 1e-6
    train_bs = 32  # 32
    valid_bs = 32  # 64
    base_model_name = "tf_efficientnet_b0_ns"
    EARLY_STOPPING = True
    DEBUG = False  # True
    EVALUATION = 'AUC'
    apex = True
    nb_workers = 6

    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float()[None, :, None, None].cuda()  # RGB
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float()[None, :, None, None].cuda()  # RG

    pooling = "max"
    pretrained = True
    num_classes = 152
    in_channels = 3
    target_columns = 'afrsil1 akekee akepa1 akiapo akikik amewig aniani apapan arcter \
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
                      towsol wantat1 warwhe1 wesmea wessan wetshe whfibi whiter whttro wiltur yebcar yefcan zebdov'.split()

    img_size = 224  # 128
    main_metric = "epoch_f1_at_03"

    period = 5
    n_mels = 224  # 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    melspectrogram_parameters = {
        "n_mels": 224,  # 128,
        "fmin": 20,
        "fmax": 16000
    }


class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = CFG.sample_rate
    duration = CFG.period
    # Melspectrogram
    n_mels = CFG.n_mels
    fmin = CFG.fmin
    fmax = CFG.fmax
