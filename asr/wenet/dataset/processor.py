# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import zipfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse
from langid.langid import LanguageIdentifier, model
import logging
import librosa
import random
import io
import boto3
import sys
import traceback
import math

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from wenet.text.base_tokenizer import BaseTokenizer

torchaudio.utils.sox_utils.set_buffer_size(16500)
from multiprocessing import Manager
AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])
# mystats = Manager().dict()

lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

logging.getLogger('langid').setLevel(logging.INFO)


class UrlOpenError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


def parse_url(elem):
    assert 'file_name' in elem
    assert 'line' in elem
    # assert isinstance(elem, dict)
    url = elem['line']
    try:
        pr = urlparse(url)
        # local file
        if pr.scheme == '' or pr.scheme == 'file':
            stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
        else:
            cmd = f'wget -q -O - {url}'
            process = Popen(cmd, shell=True, stdout=PIPE)
            elem.update(process=process)
            stream = process.stdout
        elem.update(stream=stream)
        return elem
    except Exception as ex:
        err_msg = 'Failed to open {}'.format(url)
        raise UrlOpenError(err_msg) from ex


def parse_speaker(sample, speaker_dict):
    assert 'speaker' in sample
    speaker = sample['speaker']
    sample['speaker'] = speaker_dict.get(speaker, 0)
    return sample


def detect_language(sample, limited_langs):
    assert 'txt' in sample
    # NOTE(xcsong): Because language classification may not be very accurate
    #   (for example, Chinese being classified as Japanese), our workaround,
    #   given we know for certain that the training data only consists of
    #   Chinese and English, is to limit the classification results to reduce
    #   the impact of misclassification.
    lid.set_languages(limited_langs)
    # i.e., ('zh', 0.9999999909903544)
    sample['lang'] = lid.classify(sample['txt'])[0]
    return sample


def detect_task(sample):
    # TODO(xcsong): Currently, the task is hard-coded to 'transcribe'.
    #   In the future, we could dynamically determine the task based on
    #   the contents of sample. For instance, if a sample contains both
    #   'txt_en' and 'txt_zh', the task should be set to 'translate'.
    sample['task'] = "transcribe"
    return sample




def get_rare_words(deep_bias_conf):
    rare_words = set()
    word_freqs_file = deep_bias_conf['word_freqs']
    freq_threshold = deep_bias_conf.get('freq_threshold', 20)
    with open(deep_bias_conf['word_freqs']) as f:
        word_freqs = json.load(f)
    for word, freq in word_freqs.items():
        if not word.isalpha():
            continue
        if freq <= freq_threshold:
            rare_words.add(word)
    return rare_words


def rare_utt_filter(sample, rare_words, deep_bias_conf):
    p_keep = deep_bias_conf.get('p_keep', 1)
    n_order = deep_bias_conf.get('n_order', 3)
    if True:
        rare_utt = False
        cv_terms = []
        dist_terms = []
        txt_list = sample['txt'].split()
        for word in txt_list:
            if word in rare_words:
                rare_utt = True
                word_index = txt_list.index(word)
                # uniformly choose up to n-order size
                n = random.choice(list(range(n_order)))
                if n >= len(txt_list):
                    n = 1
                if n > word_index:
                    cv_phrase = txt_list[0:word_index+1]
                else:
                    cv_phrase = txt_list[word_index - n:word_index + 1]
                cv_phrase = ' '.join(cv_phrase)
                cv_terms.append(cv_phrase)

                # dist_phrase contains one rare word and n-1 words from the utterance
                # this might be too difficult, should we pick n rare words?
                dist_phrase = random.sample(txt_list, n)
                if word in dist_phrase:
                    dist_phrase.remove(word)
                dist_phrase.append(random.choice(list(rare_words)))
                random.shuffle(dist_phrase)
                dist_phrase = ' '.join(dist_phrase)
                dist_terms.append(dist_phrase)
        if not rare_utt:
            # return None, False
            return None
        else:
            if random.random() < p_keep:
                sample['cv_list'] = cv_terms
            else:
                # don't include rare words from this utt
                sample['cv_list'] = []
            logging.debug("real: " + str(sample['cv_list']))
            sample['cv_distractors'] = dist_terms
            logging.debug("distractors: " + str(sample['cv_distractors']))
            return sample

def decode_wav(sample):
    """ Parse key/wav/txt from json line

        Args:
            sample: str, str is a json line has key/wav/txt

        Returns:
            {key, wav, sample_rate, ...}
    """
    assert 'key' in sample
    assert 'wav' in sample
    assert 'txt' in sample
    wav_file = sample['wav']
    if isinstance(wav_file, str):
        with open(wav_file, 'rb') as f:
            wav_file = f.read()
    if 'start' in sample:
        assert 'end' in sample
        sample_rate = torchaudio.info(wav_file).sample_rate
        start_frame = int(sample['start'] * sample_rate)
        end_frame = int(sample['end'] * sample_rate)
        with io.BytesIO(wav_file) as file_obj:
            waveform, _ = torchaudio.load(filepath=file_obj,
                                          num_frames=end_frame - start_frame,
                                          frame_offset=start_frame)
    else:
        with io.BytesIO(wav_file) as file_obj:
            waveform, sample_rate = torchaudio.load(file_obj)
    # del wav_file
    del sample['wav']
    sample['wav'] = waveform  # overwrite wav
    sample['sample_rate'] = sample_rate
    return sample

def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1, valid=False, track_langs=False):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    # print(f"filter = {max_output_input_ratio}")
    for sample in data:
        if not 'sample_rate' in sample:
            continue
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            mystats['minlen'] += 1
            continue
        if num_frames > max_length:
            mystats['maxlen'] += 1
            continue
        if len(sample['label']) < token_min_length:
            mystats['lbl_minlen'] += 1
            continue
        if len(sample['label']) > token_max_length:
            mystats['lbl_maxlen'] += 1
            continue

        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                mystats['min_ratio_out_over_in'] += 1
                continue
            if len(sample['label']) / num_frames > max_output_input_ratio:
                mystats['max_ratio_out_over_in'] += 1
                # print(f"reject max o/i : {sample['src']} {sample['key']} with {len(sample['label'])} labels over {num_frames} frames (len {sample['wav'].size(1)})")
                continue

        if track_langs:
            if valid:
                mystats["valid_kept"] +=1
            elif 'lang' in sample:
                if sample["lang"].decode('utf8').strip() == "en":
                    mystats["english_kept"] += 1
                elif sample["lang"].decode('utf8').strip() == "es":
                    mystats["spanish_kept"] += 1
                elif sample["lang"].decode('utf8').strip() == "fr":
                    mystats["french_kept"] += 1
            elif 'tk_lang' in sample:
                mystats["mixed_kept"] += 1
            else:
                if random.random() > 0.1:
                    mystats["notag_ignored"] += 1
                    continue
                else:
                    mystats["notag_kept"] += 1

        mystats['ok'] += 1
        yield sample


def resample(sample, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return sample


def speed_perturb(sample, speeds=None):
    """ Apply speed perturb to the sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            speeds(List[float]): optional speed

        Returns:
            key, wav, label, sample_rate}
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    speed = random.choice(speeds)
    if speed != 1.0:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        sample['wav'] = wav

    return sample


def compute_fbank(sample,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    # Only keep key, feat, label
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      sample_frequency=sample_rate)
    sample['feat'] = mat
    return sample


def sort_by_feats(sample):
    assert 'feat' in sample
    assert isinstance(sample['feat'], torch.Tensor)
    return sample['feat'].size(0)


def feats_length_fn(sample) -> int:
    assert 'feat' in sample
    return sample['feat'].size(0)


def compute_mfcc(sample,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, wav, feat, sample_rate, ...}
    """
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    mat = kaldi.mfcc(waveform,
                     num_mel_bins=num_mel_bins,
                     frame_length=frame_length,
                     frame_shift=frame_shift,
                     dither=dither,
                     num_ceps=num_ceps,
                     high_freq=high_freq,
                     low_freq=low_freq,
                     sample_frequency=sample_rate)
    sample['feat'] = mat
    return sample


def compute_log_mel_spectrogram(sample,
                                n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav'].squeeze(0)  # (channel=1, sample) -> (sample,)
    if padding > 0:
        waveform = F.pad(waveform, (0, padding))
    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform,
                      n_fft,
                      hop_length,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    sample['feat'] = log_spec.transpose(0, 1)
    return sample


def tokenize(sample, tokenizer: BaseTokenizer):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            sample: {key, wav, txt, sample_rate, ...}

        Returns:
            {key, wav, txt, tokens, label, sample_rate, ...}
    """
    assert 'txt' in sample
    tokens, label = tokenizer.tokenize(sample['txt'])
    sample['tokens'] = tokens
    sample['label'] = label
    return sample

def tokenize_cv_list(sample, tokenizer: BaseTokenizer):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            sample: {key, wav, txt, sample_rate, ...}

        Returns:
            {key, wav, txt, tokens, label, sample_rate, ...}
    """
    assert 'cv_list' in sample
    assert 'cv_distractors' in sample
    cv_tokens_list = []
    cv_label_list = []
    dist_tokens_list = []
    dist_label_list = []
    for word in sample['cv_list']:
                cv_tokens, cv_label = tokenizer.tokenize(word)
                cv_tokens_list.append(cv_tokens)
                cv_label_list.append(cv_label)
    sample['cv_tokens_list'] = cv_tokens_list
    sample['cv_label_list'] = cv_label_list

    for dist in sample['cv_distractors']:
        dist_tokens, dist_label = tokenizer.tokenize(dist)
        dist_tokens_list.append(dist_tokens)
        dist_label_list.append(dist_label)
    sample['dist_tokens_list'] = dist_tokens_list
    sample['dist_label_list'] = dist_label_list

    return sample


def filter(sample,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            sample: {key, wav, label, sample_rate, ...}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            bool: True to keep, False to filter
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    # sample['wav'] is torch.Tensor, we have 100 frames every second
    num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
    if num_frames < min_length:
        return False
    if num_frames > max_length:
        return False

    if 'label' in sample:
        if len(sample['label']) < token_min_length:
            return False
        if len(sample['label']) > token_max_length:
            return False
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                return False
            if len(sample['label']) / num_frames > max_output_input_ratio:
                return False
    return True


def spec_aug(sample, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            sample: {key, feat, ...}
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            {key, feat, ....}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    max_freq = y.size(1)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for _ in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    sample['feat'] = y
    return sample


def spec_sub(sample, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            sample: Iterable{key, feat, ...}
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            {key, feat, ...}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    for _ in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = x[start - pos:end - pos, :]
    sample['feat'] = y
    return sample


def spec_trim(sample, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            sample: {key, feat, label}
            max_t: max width of length trimming

        Returns:
            {key, feat, label}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    max_frames = x.size(0)
    length = random.randint(1, max_t)
    if length < max_frames / 2:
        y = x.clone().detach()[:max_frames - length]
        sample['feat'] = y
    return sample

def set_epoch(new_epoch):
    global global_epoch
    global_epoch = new_epoch


def get_epoch():
    return global_epoch


def filter_cv_by_epoch(cv_terms, deep_bias_conf):
    # Increase total number of CV terms over time
    total = len(cv_terms)
    epoch = get_epoch()
    max_epoch = deep_bias_conf.get('max_epoch', 10)
    target_len = min(total, math.ceil(total * (epoch + 1) / (max_epoch + 1)))
    logging.debug(f'processor filter_cv_by_epoch | epoch: {epoch}, total terms: {total}, target_len: {target_len}')
    return random.sample(cv_terms, target_len)


def batch_cv_list(sample, deep_bias_conf):
    assert isinstance(sample, list)
    # Concatenate all the cv_terms for the batch together into one list
    cv_terms = [s['cv_label_list'] for s in sample]
    distractor_terms = [s['dist_label_list'] for s in sample]
    batch_cv_terms = [tuple(term) for term_list in cv_terms for term in term_list]
    batch_dist_terms = [tuple(term) for term_list in distractor_terms for term in term_list]
    # Add some proportion of distractor terms
    dist_ratio = deep_bias_conf.get('distractor_ratio', 0.2)
    num_dist = round(len(batch_dist_terms)*dist_ratio)
    batch_terms = batch_cv_terms + random.sample(batch_dist_terms, num_dist)
    batch_terms = filter_cv_by_epoch(batch_terms, deep_bias_conf)
    assert isinstance(batch_terms, list)
    return batch_terms


def padding(data, pass_cat_emb=False, deep_biasing_conf={}):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    
    #logging.info(f"padding: with pass_cat_emb {pass_cat_emb}")

    assert isinstance(sample, list)
    feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    feats_lengths = torch.tensor([sample[i]['feat'].size(0) for i in order],
                                 dtype=torch.int32)
    sorted_feats = [sample[i]['feat'] for i in order]
    sorted_keys = [sample[i]['key'] for i in order]
    sorted_labels = [
        torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
    ]
    sorted_wavs = [sample[i]['wav'].squeeze(0) for i in order]
    langs = [sample[i]['lang'] for i in order]
    tasks = [sample[i]['task'] for i in order]
    #for x in sorted_labels: print(f"labels : {x}")

    label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                 dtype=torch.int32)
    wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                               dtype=torch.int32)
    padded_feats = pad_sequence(sorted_feats,
                                batch_first=True,
                                padding_value=0)
    padding_labels = pad_sequence(sorted_labels,
                                  batch_first=True,
                                  padding_value=-1)
    padded_wavs = pad_sequence(sorted_wavs, batch_first=True, padding_value=0)


    batch = {
        "keys": sorted_keys,
        "feats": padded_feats,
        "target": padding_labels,
        "feats_lengths": feats_lengths,
        "target_lengths": label_lengths,
        "pcm": padded_wavs,
        "pcm_length": wav_lengths,
        "langs": langs,
        "tasks": tasks,
    }

    if pass_cat_emb:
        assert 'cat_emb' in sample[0]
        sorted_cat_emb = [sample[i]['cat_emb'] for i in order]
        padded_cat_emb = pad_sequence(sorted_cat_emb, batch_first=True, padding_value=0)
        batch['cat_embs'] = padded_cat_emb

    if 'speaker' in sample[0]:
        speaker = torch.tensor([sample[i]['speaker'] for i in order],
                               dtype=torch.int32)
        batch['speaker'] = speaker

    if 'cv_list' in sample[0]:
        batch_cv_terms = batch_cv_list(sample, deep_biasing_conf)
        batch_cv_terms = [torch.tensor(list(term), dtype=torch.int64) for term in batch_cv_terms]
        padded_cv_terms = pad_sequence(batch_cv_terms, batch_first=True, padding_value=0)
        cv_lengths = torch.tensor([x.size(0) for x in batch_cv_terms])
        batch['cv_list'] = padded_cv_terms
        batch['cv_list_lengths'] = cv_lengths

    return batch


class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        return False
