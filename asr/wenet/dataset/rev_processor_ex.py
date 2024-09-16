# Copyright (c) 2021 Rev.com. (author: Jp Robichaud)
#
import logging
from multiprocessing import cpu_count
import os
import random
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import time
from numpy import random as nprnd
from timeit import default_timer as timer
from multiprocessing import Manager
AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

# trying to set a random seed different for each process
nprnd.seed((os.getpid() * int(time.time())) % 123456789)

mystats = Manager().dict()
mystats['reject_1'] = 0
mystats['reject_2'] = 0
mystats['reject_5'] = 0
mystats['reject_9'] = 0
mystats['excluded_by_key'] = 0
mystats['ok'] = 0

# from torchaudio data augmentation tutorial
def get_rir_sample(RIR_PATH, resample=None, processed=False):
  rir_raw, sample_rate = _get_sample(RIR_PATH, resample=resample)
  if not processed:
    return rir_raw, sample_rate
  rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
  rir = rir / torch.norm(rir, p=2)
  rir = torch.flip(rir, [1])
  return rir, sample_rate 

# from torchaudio data augmentation tutorial
def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


#_ne -ef  techically, impuse file list should come from the model yaml configuration
# same with prob
def apply_rir(data, sample_rate=16000, prob=0.2, impulse_list_fn = None):
    """ Apply room impulse response to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: resample rate at which the impulse response is sampled
            prob: probability of applying rir
            impulse_list_fn: file containing the list of impulse response files

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """

    ## TODO : technically, we could keep the orignal wav data in a separate field
    ## and add a loss term to try to make the encoder output the same between the original
    ## and the augmented data

    # this will reload the RIR filters each epoch, but this is really quick anyways
    on_gpu = torch.cuda.is_available()
    # on_gpu = False
    impulse_rir = []
    if impulse_list_fn is not None:
        with open(impulse_list_fn, "r") as f:
            for line in f:
                rir_wav_fn = line.strip()
                if not os.path.exists(rir_wav_fn):
                    logging.warning(f'RIR wav file {rir_wav_fn} does not exist, skipping!')
                    continue
                try:
                    rir_tensor, sr = get_rir_sample(rir_wav_fn, resample=sample_rate, processed=True)
                    if on_gpu:
                        rir_tensor = rir_tensor.to(device='cuda')
                    impulse_rir.append(rir_tensor)
                except BaseException as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    logging.warning(f'Exception caught while attemping to prepare RIR wav file {rir_wav_fn} does not exist, skipping!')
                    continue


    if len(impulse_rir) == 0:
        logging.warning('RIR wav list was none or empty, no RIR will be applied')
        for sample in data:
            yield sample
        return

    # logging.info(f"Applying RIR with a probability of {prob}")
    rir_len = len(impulse_rir)
    for sample in data:
        t0 = timer()
        wav_sample_rate = sample['sample_rate']
        waveform = sample['wav']
        # logging.info(f"Applying SR vs WSR {sample_rate} vs {wav_sample_rate}")
        assert sample_rate == wav_sample_rate

        if random.random() > prob:
            # logging.info(f"Applying NO-RIR")
            t1 = timer()
            e = (t1 - t0)
            sample['rir_elapsed'] = e
            yield sample
            continue

        waveform = sample['wav']
        # ok, we want to apply rir, let's pick one
        rir = random.choice(impulse_rir)
        # rir = nprnd.choice(impulse_rir)
        # rir = impulse_rir[nprnd.randint(rir_len)]
        # logging.info(f"Applying RIR")
        if on_gpu:
            waveform = waveform.to(device='cuda')

        speech_ = torch.nn.functional.pad(waveform, (rir.shape[1]-1, 0))
        augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]
        if on_gpu:
            augmented = augmented.cpu()

        sample['wav'] = augmented
        t1 = timer()
        e = (t1 - t0)
        sample['rir_elapsed'] = e
        yield sample


def apply_telephony(data, prob=0.2):
    """ Apply telephony effects to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            prob: probability of applying telephony effects

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    codec_configs = [
        ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8bit_mu-law"),
        ({"format": "amb"}, "AMB"),

        # FIXME: These two below throw messages to stdout or stderr, let's avoid them for now.
        # ({"format": "gsm"}, "GSM-FR"),
        # ({"format": "amr-nb"}, "AMR-NB"),
    ]

    ## TODO : technically, we could keep the orignal wav data in a separate field
    ## and add a loss term to try to make the encoder output the same between the original
    ## and the augmented data

    # logging.info(f"Applying telephony of {prob}")
    for sample in data:
        t0 = timer()
        wav_sample_rate = sample['sample_rate']
        waveform = sample['wav']

        if random.random() > prob:
            # logging.info(f"Applying codec ORIGINAL")
            t1 = timer()
            e = (t1 - t0)
            sample['elapsed'] = e
            yield sample
            continue

        # ok, we want to apply rir, let's pick one
        waveform = sample['wav']
        speech, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            wav_sample_rate,
            effects=[
                ["lowpass", f"{4000 - random.randint(-200, 200)}"],
                ["compand", "0.02,0.05",
                    "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
                ["rate", "8000"],
            ],
        )
        t1 = timer()
        e = (t1 - t0)
        sample['elapsed_p1'] = e

        c = random.randint(0, len(codec_configs)-1)
        codec_type = codec_configs[c][0]
        # logging.info(f"Applying codec {codec_type['format']}")
        speech = F.apply_codec(speech, sample_rate, **codec_type)
        t2 = timer()
        e = (t2 - t1)
        sample['elapsed_p2'] = e
        resampled_waveform = F.resample(
            speech,
            8000,
            wav_sample_rate,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="kaiser_window",
            beta=8.555504641634386
        )

        t3 = timer()
        e = (t3 - t2)
        sample['elapsed_p3'] = e

        sample['wav'] = resampled_waveform
        t4 = timer()
        e = (t4 - t0)
        sample['elapsed'] = e

        yield sample

def apply_telephony_gpu(data, prob=0.2):
    """ Apply telephony effects to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            prob: probability of applying telephony effects

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    codec_configs = [
        ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8bit_mu-law"),
        ({"format": "amb"}, "AMB"),

        # FIXME: These two below throw messages to stdout or stderr, let's avoid them for now.
        # ({"format": "gsm"}, "GSM-FR"),
        # ({"format": "amr-nb"}, "AMR-NB"),
    ]
    on_gpu = torch.cuda.is_available()

    wav_sample_rate = 16000
    transforms_list = []
    for low_freq in [8000 - i for i in [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]]:
        transforms = torch.nn.Sequential(
            T.Resample(
                wav_sample_rate,
                low_freq,
                lowpass_filter_width=16, # default 6
                rolloff=0.85,
                resampling_method="sinc_interpolation",
            ),
            T.MuLawEncoding(256),
            T.MuLawDecoding(256),
            T.Resample(
                low_freq,
                wav_sample_rate,
                lowpass_filter_width=16, # default 6
                rolloff=0.85,
                resampling_method="sinc_interpolation",
            ),
        ).to(device='cuda')
        transforms_list.append(transforms)

    # logging.info(f"Applying telephony of {prob}")
    for sample in data:
        t0 = timer()
        if random.random() > prob:
            # logging.info(f"Applying codec ORIGINAL")
            t1 = timer()
            e = (t1 - t0)
            sample['elapsed'] = e
            yield sample
            continue

        # ok, we want to apply rir, let's pick one
        wav_sample_rate = sample['sample_rate']

        # sox_effects doesn't work on GPU
        # speech, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        #     waveform,
        #     wav_sample_rate,
        #     effects=[
        #         ["lowpass", f"{4000 - random.randint(-200, 200)}"],
        #         ["compand", "0.02,0.05",
        #             "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
        #         ["rate", "8000"],
        #     ],
        # )

        if on_gpu:
            waveform = sample['wav'].to(device='cuda')
        else:
            waveform = sample['wav']

        # resampled_waveform = F.lowpass_biquad(
        #     resampled_waveform,
        #     8000,
        #     cutoff_freq=4000 - random.randint(-200, 200)
        # )

        transforms = random.choice(transforms_list)
        resampled_waveform = transforms(waveform)

        # back on CPU
        sample['wav'] = resampled_waveform.cpu()
        t4 = timer()
        e = (t4 - t0)
        sample['elapsed'] = e

        yield sample


def filter_wordy(data):
    """ Filter sample according to duration and word per second.
        Inplace operation.

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    # print(f"filter = {max_output_input_ratio}")
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'txt' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        dur = sample['wav'].size(1) / sample['sample_rate']

        # hotfix, removing <sw>
        txt = sample['txt'].replace("<sw> "," ")

        nwds = len(txt.split())
        wps = nwds / dur
        if dur <= 1:
            if wps > 5:
                mystats['reject_1'] += 1
                continue
        elif dur <= 2:
            if wps > 8:
                mystats['reject_2'] += 1
                continue
        elif dur <= 5:
            if wps > 6:
                mystats['reject_5'] += 1
                continue
        elif wps > 5:
            mystats['reject_9'] += 1
            continue
        mystats['ok'] += 1
        yield sample

def exclude_keys(data, exclude_fn):
    to_exclude = set()

    with open(exclude_fn, 'r') as reader:
        for line in reader:
            line = line.strip()
            to_exclude.add(line)

    logging.info(f"We have {len(to_exclude)} utterances in the exclusion list")
    for sample in data:
        assert 'key' in sample
        key = sample['key']
        if key in to_exclude:
            mystats['excluded_by_key']+= 1
            continue
        yield sample

