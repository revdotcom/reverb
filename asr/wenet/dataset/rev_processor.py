# Copyright (c) 2021 Rev.com. (author: Jp Robichaud)
#
import logging
import os
import random
import torch
import torchaudio
import torchaudio.functional as F
import time
from multiprocessing import Manager
from numpy import random as nprnd

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])
mystats = Manager().dict()
mystats_longyeahokay = Manager().dict()

DEFAULTS_VALS={'lang' : 'en', 'style' : 'nv'}


# trying to set a random seed different for each process
nprnd.seed((os.getpid() * int(time.time())) % 123456789)

# JPR with the new wenet approach, these methods are getting called with the sample object
# so filter methods should just return true/false
# false : reject the sample
# true : keep the sample
def random_skip_non_sw(sample, random_skip_non_sw):

    if 'skip_non_sw' not in mystats:
        mystats['skip_non_sw'] = 0

    assert 'txt' in sample
    txt = sample['txt']

    if '<sw>' in txt and random.random() < random_skip_non_sw:
        mystats['skip_non_sw'] += 1
        return False

    yield True

def add_one_hot(sample, emb_len=1, field='lang', one_hot_ids=None, multi_hot=False, force_hot=None, defaults_vals=DEFAULTS_VALS):
    """ add language information - currently 1-hot vector, but could have other options in the future

            inside dataset_conf in .yaml config file, added optional arguments for this function. Example:

            add_cat_emb: true
            cat_emb_conf:
            field: 'lang'
            emb_len: 3
            one_hot_ids:
                en: 0
                es: 1
                fr: 2

            Args:
                data: Iterable[{key, feat, label}]

            Returns
                Iterable[{key, feat, label}]
    """
    assert 'feat' in sample

    x = sample['feat']
    assert isinstance(x, torch.Tensor)

    y = x.clone().detach()
    max_frames = y.size(0)
    # create features that will be added to every frame
    onehot = torch.zeros(max_frames, emb_len)

    if field == 'lang' and not field in sample:
        if 'tk_lang' in sample:
            sample[field] = sample['tk_lang']
        else:
            # large (e.g. 50K or 100K hours) English data is a pain to add the "lang" tag to,
            # so for now, just don't, and assume untagged data is English
            sample[field]=defaults_vals['lang'] #'en'
    elif field == 'style' and not field in sample:
        # sample[field] = 'nv'
        sample[field] = defaults_vals['style'] #'nv

    if field in sample and one_hot_ids is not None:
        # in theory, could add different fields to the shards that could hold different types of
        # data, once we get into this function we know that the specified field in the shard should
        # contain string data that needs to be decoded
        try:
            sample[field] = sample[field].decode('utf8').strip()
        except:
            pass
        for f in sample[field].split():
            # set specified feature to 1
            onehot[:, one_hot_ids[f]] = 1.

    # force other features to 1, if specified
    if force_hot is not None:
        for f in force_hot:
            onehot[:, int(f)] = 1.

    # sample random multi-hot vector, if specified
    if multi_hot:
        if random.random() > 0.75:
            samp = random.randint(0, emb_len)
            if samp == emb_len:
                onehot = torch.ones(max_frames, emb_len)
            else:
                onehot[:, samp] = 1.

    # normalize embedding
    onehot = onehot / onehot[0, :].sum()

    sample['feat'] = torch.cat((y, onehot), 1)

    return sample

def pass_one_hot(sample, emb_len=1, field='lang', one_hot_ids=dict(), multi_hot=False, force_hot=[], defaults_vals=DEFAULTS_VALS):

    """ add language information - currently 1-hot vector, but could have other options in the future

        Args:
            data: Iterable[{key, feat, label}]

        Returns
            Iterable[{key, feat, label}]
    """
    onehot = torch.zeros(emb_len,)
    if field == 'lang' and not field in sample:
        if 'tk_lang' in sample:
            sample[field] = sample['tk_lang'].decode('utf8').strip()
        else:
            # large (e.g. 50K or 100K hours) English data is a pain to add the "lang" tag to,
            # so for now, just don't, and assume untagged data is English
            # sample[field]='en'
            sample[field]=defaults_vals['lang'] #'en'
    elif field == 'style' and not field in sample:
        # sample[field] = 'nv'
        sample[field] = defaults_vals['style'] #'nv

    for f in sample[field].split():
        onehot[one_hot_ids[f]] = 1.
    
    # force other features to 1, if specified
    for f in force_hot:
        onehot[int(f)] = 1.

    # sample random multi-hot vector, if specified
    if multi_hot:
        if random.random() > 0.75:
            samp = random.randint(0, emb_len)
            if samp == emb_len:
                onehot = torch.ones(emb_len)
            else:
                onehot[samp] = 1.

    # normalize embedding
    onehot = onehot / onehot[:].sum()

    sample['cat_emb'] = onehot

    return sample

class SpecialTokensHandler:
    def __init__(self, config):
        self.reject_set = set()
        self.remove_set = set()
        self.relabel_map = dict()
        self.remove_trailing_dash = config.get('remove_trailing_dash', False)
        self.verbose = config.get('verbose', False)

        if 'reject_on' in config:
            for tk in config['reject_on']:
                self.reject_set.add(tk)
                if tk not in mystats:
                    mystats[tk] = 0

        if 'remove' in config:
            for tk in config['remove']:
                self.remove_set.add(tk)
                if tk not in mystats:
                    mystats[tk] = 0

        if 'relabel' in config:
            for tk, tk_dest in config['relabel']:
                self.relabel_map[tk] = tk_dest
                if tk not in mystats:
                    mystats[tk] = 0

    def filter(self, sample):
        if sample is None:
            return False
        return True

    def transform(self, sample):
        assert 'txt' in sample
        txt = sample['txt']
        words = txt.split()
        new_words = []
        should_reject = False

        for w in words:
            if self.remove_trailing_dash and w.endswith('-'):
                w = w[:-1]

            if w in self.reject_set:
                should_reject = True
                mystats[w] += 1
                break

            if w in self.remove_set:
                mystats[w] += 1
                continue

            if w in self.relabel_map:
                mystats[w] += 1
                new_words.append(self.relabel_map[w])
            else:
                new_words.append(w)

        if should_reject or len(new_words) == 0:
            return None
        
        sample['otxt'] = txt
        sample['txt'] = " ".join(new_words)
        if self.verbose and sample['txt'] != sample['otxt']:
             print(f"Before: {sample['otxt']} || After: {sample['txt']}")

        return sample

    def print_stats(self):
        print(mystats)


def handle_special_tokens(data, config):
    reject_set = set()
    remove_set = set()
    relabel_map = dict()

    remove_trailing_dash = config.get('remove_trailing_dash', False)
    if 'reject_on' in config:
        for tk in config['reject_on']:
            reject_set.add(tk)
            if tk not in mystats:
                mystats[tk] = 0

    if 'remove' in config:
        for tk in config['remove']:
            remove_set.add(tk)
            if tk not in mystats:
                mystats[tk] = 0

    if 'relabel' in config:
        for tk, tk_dest in config['relabel']:
            relabel_map[tk] = tk_dest
            if tk not in mystats:
                mystats[tk] = 0


    for sample in data:
        assert 'txt' in sample
        txt = sample['txt']
        words = txt.split()
        new_words = []
        should_reject = False

        for w in words:
            if remove_trailing_dash and w.endswith('-'):
                w = w[:-1]

            if w in reject_set:
                should_reject = True
                mystats[w] += 1
                break

            if w in remove_set:
                mystats[w] += 1
                continue

            if w in relabel_map:
                mystats[w] += 1
                new_words.append(relabel_map[w])
            else:
                new_words.append(w)

        if should_reject or len(new_words) == 0:
            continue
        
        sample['otxt'] = txt
        sample['txt'] = " ".join(new_words)
        #if sample['txt'] != sample['otxt']:
        #     print(f"Before: {sample['otxt']} || After: {sample['txt']}")
        yield sample

    print(mystats)


def generate_speaker_switch_utterances(data, config):
    """ Generate speaker switch utterances.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            config: dict
                'prob': probability of applying speaker switch
                'speaker_list_fn': file containing the list of speakers

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """

    def get_speaker_id(key):
        # TC00000000P-1.aligned.wav_speaker00000-TC00000000P-1-A-00001
        if '-' in key:
            return key[:key.rindex('-')]     
        return key


    # FIXME : take these from yaml config
    sampling_rate = 16000
    min_audio_len_acceptable_secs = 1 # 1 seconds, if audio is smaller than this, skip the agglomeration steps
    min_audio_len_secs = 10 # 10 seconds, if audio is already greater than this, we leave it alone
    max_audio_len_secs = 20 # 20 seconds
    max_utt_combined = 7

    epoch = data.source.get_epoch()
    min_epoch = config.get("enable_after_epoch", -1)
    if min_epoch >= epoch:
        # skip this processor if the epoch isn't above the requested threshold
        for sample in data:
            yield sample
        return

    curr_speaker = None
    curr_sample = None
    num_utt_combined = 0
    for sample in data:
        key = sample['key']
        spk = get_speaker_id(key)
        if curr_speaker is None:
            curr_speaker = spk
            curr_sample = sample
            num_utt_combined = 1
            continue

        # the utterance is too small to be useable in composition
        if curr_sample['wav'].size(1) < sampling_rate * min_audio_len_acceptable_secs:
            yield curr_sample
            curr_sample = sample
            curr_speaker = spk
            num_utt_combined = 1
            continue

        # the utterance already has enough data
        if curr_sample['wav'].size(1) > sampling_rate * min_audio_len_secs:
            yield curr_sample
            curr_sample = sample
            curr_speaker = spk
            num_utt_combined = 1
            continue

        # combining audios would bump over the limit
        if num_utt_combined >= max_utt_combined or (curr_sample['wav'].size(1) + sample['wav'].size(1)) > sampling_rate * max_audio_len_secs:
            yield curr_sample
            curr_sample = sample
            curr_speaker = spk
            num_utt_combined = 1
            continue

        # merging
        num_utt_combined += 1
        d0 = curr_sample['wav'].size(1)
        d1 = sample['wav'].size(1)
        curr_sample['wav'] = torch.cat([curr_sample['wav'], sample['wav']], dim=1)
        d2 = curr_sample['wav'].size(1)
        sep = " " if curr_speaker == spk else " <sw> "
        # logging.info(f"Combining {curr_sample['key']} and {sample['key']}, ({d0} + {d1} = {d2} samples, {curr_sample['wav'].shape}) giving transcription : {curr_sample['txt']}/{sep}/{sample['txt']}")
        curr_sample['txt'] = (curr_sample['txt']  + sep + sample['txt']).replace("<sw> <sw>", "<sw>")
        curr_speaker = spk

        # this can't work "in general"
        # for k in sample.keys():
        #     if k in ["wav", "spk", "txt", "key"]:
        #         continue
        #     curr_sample[k] += sample[k]
        
    yield curr_sample


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


class RIREngine:
    def __init__(self, config):
        self.config = config
        # print(f"ROR CONFIG : {config}")
        # sample_rate=16000, prob=0.2, impulse_list_fn = None):
        self.impulse_rir= []
        self.sample_rate = config.get('sample_rate', 16000)
        self.prob = config.get('prob', 0.2)
        # self.on_gpu = False
        self.on_gpu = True
        self.impulse_list_fn = config.get('impulse_list_fn', "dummy")
        self.local_random = random.Random()
        self.load_rirs(self.impulse_list_fn)

    def load_rirs(self, impulse_list_fn):
        with open(impulse_list_fn, "r") as f:
            for line in f:
                rir_wav_fn = line.strip()
                if not os.path.exists(rir_wav_fn):
                    logging.warning(f'RIR wav file {rir_wav_fn} does not exist, skipping!')
                    continue
                try:
                    rir_tensor, sr = get_rir_sample(rir_wav_fn, resample=self.sample_rate, processed=True)
                    if self.on_gpu:
                        waveform = waveform.to(device='cuda', non_blocking=True)
                    self.impulse_rir.append(rir_tensor)
                except BaseException as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    logging.warning(f'Exception caught while attemping to prepare RIR wav file {rir_wav_fn} does not exist, skipping!')
                    continue

    def apply_rir(self, sample):
        # if on_gpu and (sample_id>0 and sample_id % 10000 == 0):
            # torch.cuda.empty_cache()

        if self.local_random.random() > self.prob:
            # logging.info(f"Applying NO-RIR")
            return sample

        wav_sample_rate = sample['sample_rate']
        waveform = sample['wav']
        # ok, we want to apply rir, let's pick one
        rir = self.local_random.choice(self.impulse_rir)
        # rir = nprnd.choice(impulse_rir)
        # rir = impulse_rir[nprnd.randint(rir_len)]
        # logging.info(f"Applying RIR")
        if self.on_gpu:
            waveform = waveform.to(device='cuda', non_blocking=True)

        waveform = torch.nn.functional.pad(waveform, (rir.shape[1]-1, 0))
        augmented = torch.nn.functional.conv1d(waveform[None, ...], rir[None, ...])[0]
        if self.on_gpu:
            augmented = augmented.cpu()

        del waveform
        sample['wav'] = augmented
        return sample

# This happens on the CPU as sox_effects are CPU only
def apply_telephony(sample, prob=0.2):
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
    if random.random() > prob:
        # logging.info(f"Applying codec ORIGINAL")
        return sample

    # ok, we want to apply rir, let's pick one
    # print(sample)
    wav_sample_rate = sample['sample_rate']
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

    c = random.randint(0, len(codec_configs)-1)
    codec_type = codec_configs[c][0]
    # logging.info(f"Applying codec {codec_type['format']}")
    speech = F.apply_codec(speech, sample_rate, **codec_type)
    # resampled_waveform = F.resample(
    #     speech,
    #     8000,
    #     wav_sample_rate,
    #     lowpass_filter_width=16,
    #     rolloff=0.85,
    #     resampling_method="sinc_interp_kaiser",
    #     beta=8.555504641634386
    # )
    resampled_waveform = F.resample(
        speech,
        8000,
        wav_sample_rate,
        lowpass_filter_width=16,
        rolloff=0.85,
        # resampling_method="sinc_interp_kaiser",
        # beta=8.555504641634386
    )

    sample['wav'] = resampled_waveform
    return sample


def filter_long_yeah_okay(data, too_long_duration=1.5, word_subset=None):
    """Filter single word samples that are too long.
    Ported from revspeech/kaldi_egs/utils/remove_long_yeah_okay_segments.pl

    Args::
        data: Iterable[{key, wav, label, sample_rate}]
        too_long_duration (float): drop single word utterances over this duration
        word_subset (List[str]): only drop single word utterances in this list of words

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    mystats_longyeahokay['##total_utterances_filtered##'] = 0
    mystats_longyeahokay['##total_duration_filtered##'] = 0.0

    for sample in data:
        if not 'sample_rate' in sample:
            continue
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample

        dur = sample['wav'].size(1) / sample['sample_rate']
        num_w = len(sample['txt'].split())

        if num_w == 1 and dur >= too_long_duration:
            if not word_subset:
                if sample['txt'] in mystats_longyeahokay:
                    mystats_longyeahokay[sample['txt']] += 1
                else:
                    mystats_longyeahokay[sample['txt']] = 1

                mystats_longyeahokay['##total_utterances_filtered##'] += 1
                mystats_longyeahokay['##total_duration_filtered##'] += dur

                continue
            elif sample['txt'] in word_subset:
                if sample['txt'] in mystats_longyeahokay:
                    mystats_longyeahokay[sample['txt']] += 1
                else:
                    mystats_longyeahokay[sample['txt']] = 1

                mystats_longyeahokay['##total_utterances_filtered##'] += 1
                mystats_longyeahokay['##total_duration_filtered##'] += dur

                continue

        yield sample
