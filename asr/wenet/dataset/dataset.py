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

from functools import partial
from typing import Optional
from wenet.dataset import processor
from wenet.dataset import rev_processor
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.utils.file_utils import read_symbol_table
import sys
import logging


def Dataset(data_type,
            data_list_file,
            tokenizer: Optional[BaseTokenizer] = None,
            conf=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer or None): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert conf is not None
    assert data_type in ['raw', 'shard']
    # cycle dataset
    cycle = conf.get('cycle', 1)
    # stage1 shuffle: source
    list_shuffle = conf.get('list_shuffle', True)
    list_shuffle_size = sys.maxsize
    if list_shuffle:
        list_shuffle_conf = conf.get('list_shuffle_conf', {})
        list_shuffle_size = list_shuffle_conf.get('shuffle_size',
                                                  list_shuffle_size)

    # if data_type == 'raw':
    #     dataset = WenetRawDatasetSource(data_list_file, partition=partition)
    #     dataset = dataset.map(processor.parse_json)
    # else:
    #     dataset = WenetTarShardDatasetSource(data_list_file,
    #                                          partition=partition)
    if data_type == 'raw':
        dataset = WenetRawDatasetSource(data_list_file,
                                        partition=partition,
                                        shuffle=list_shuffle,
                                        shuffle_size=list_shuffle_size,
                                        cycle=cycle)
        dataset = dataset.map(processor.parse_json)
    else:
        dataset = WenetTarShardDatasetSource(data_list_file,
                                             partition=partition,
                                             shuffle=list_shuffle,
                                             shuffle_size=list_shuffle_size,
                                             cycle=cycle)
    dataset = dataset.map_ignore_error(processor.decode_wav)

    speaker_conf = conf.get('speaker_conf', None)
    if speaker_conf is not None:
        assert 'speaker_table_path' in speaker_conf
        speaker_table = read_symbol_table(speaker_conf['speaker_table_path'])
        dataset = dataset.map(
            partial(processor.parse_speaker, speaker_dict=speaker_table))

    deep_biasing = conf.get('deep_bias_conf', {}).get('deep_biasing', False)
    deep_biasing_conf = conf.get('deep_bias_conf',{})
    if deep_biasing:
        rare_words = processor.get_rare_words(conf['deep_bias_conf'])
        dataset = dataset.map(processor.rare_utt_filter, rare_words, deep_biasing_conf)
        dataset = dataset.map(partial(processor.tokenize_cv_list, tokenizer=tokenizer))

    if tokenizer is not None:
        dataset = dataset.map(partial(processor.tokenize, tokenizer=tokenizer))

    filter_conf = conf.get('filter_conf', {})
    dataset = dataset.filter(partial(processor.filter, **filter_conf))

    handle_special_tokens = conf.get('handle_special_token', False)
    handle_special_tokens_conf = conf.get('handle_special_token_conf', {})
    if handle_special_tokens:
        special_tk_handler = rev_processor.SpecialTokensHandler(handle_special_tokens_conf)
        dataset = dataset.map(special_tk_handler.transform)
        dataset = dataset.filter(special_tk_handler.filter)


    resample_conf = conf.get('resample_conf', {})
    dataset = dataset.map(partial(processor.resample, **resample_conf))

    speed_perturb = conf.get('speed_perturb', False)
    speed_perturb_conf = conf.get('speed_perturb_conf', {})
    if speed_perturb:
        dataset = dataset.map(partial(processor.speed_perturb))

    if conf.get('apply_telephony', False) and 'apply_telephony_conf' in conf:
        print("telephony is ON")
        telephony_conf = conf.get('apply_telephony_conf', {})
        dataset = dataset.map(partial(rev_processor.apply_telephony, **telephony_conf))
    else:
        print("telephony is OFF")

    if conf.get('apply_rir', False) and 'apply_rir_conf' in conf:
        print("RIR is ON")
        rir_engine = rev_processor.RIREngine(conf.get('apply_rir_conf', {}))
        dataset = dataset.map(rir_engine.apply_rir)
        # rir_conf = conf.get('apply_rir_conf', {})
        # dataset = dataset.map(partial(rev_processor.apply_rir, **rir_conf))
    else:
        print("RIR is OFF")

    feats_type = conf.get('feats_type', 'fbank')
    assert feats_type in ['fbank', 'mfcc', 'log_mel_spectrogram']
    if feats_type == 'fbank':
        fbank_conf = conf.get('fbank_conf', {})
        dataset = dataset.map(partial(processor.compute_fbank, **fbank_conf))
    elif feats_type == 'mfcc':
        mfcc_conf = conf.get('mfcc_conf', {})
        dataset = dataset.map(partial(processor.compute_mfcc, **mfcc_conf))
    elif feats_type == 'log_mel_spectrogram':
        log_mel_spectrogram_conf = conf.get('log_mel_spectrogram_conf', {})
        dataset = dataset.map(
            partial(processor.compute_log_mel_spectrogram,
                    **log_mel_spectrogram_conf))
    spec_aug = conf.get('spec_aug', True)
    spec_sub = conf.get('spec_sub', False)
    spec_trim = conf.get('spec_trim', False)
    spec_trim = conf.get('spec_trim', False)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = dataset.map(partial(processor.spec_aug, **spec_aug_conf))
    if spec_sub:
        spec_sub_conf = conf.get('spec_sub_conf', {})
        dataset = dataset.map(partial(processor.spec_sub, **spec_sub_conf))
    if spec_trim:
        spec_trim_conf = conf.get('spec_trim_conf', {})
        dataset = dataset.map(partial(processor.spec_trim, **spec_trim_conf))

    language_conf = conf.get('language_conf', {"limited_langs": ['zh', 'en']})
    dataset = dataset.map(partial(processor.detect_language, **language_conf))
    dataset = dataset.map(processor.detect_task)

    add_cat_emb = conf.get('add_cat_emb', False)
    cat_emb_conf = conf.get('cat_emb_conf', {})
    if add_cat_emb:
        # note: this is currently only one-hot embeddings
        dataset = dataset.map(partial(rev_processor.add_one_hot, **cat_emb_conf))

    pass_cat_emb = conf.get('pass_cat_emb', False)
    if pass_cat_emb:
        logging.info("passing category embedding to asr model")
        cat_emb_conf = conf.get('cat_emb_conf', {})
        # note: this is currently only one-hot embeddings
        dataset = dataset.map(partial(rev_processor.pass_one_hot, **cat_emb_conf))

    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
                               key_func=processor.sort_by_feats)

    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    print(f"{batch_type}")
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        # dataset = dataset.batch(batch_size, wrapper_class=processor.padding)
        dataset = dataset.batch(batch_size, wrapper_class=partial(processor.padding,pass_cat_emb=pass_cat_emb, deep_biasing_conf=deep_biasing_conf))
    elif batch_type == 'bucket':
        assert 'bucket_boundaries' in batch_conf
        assert 'bucket_batch_sizes' in batch_conf
        dataset = dataset.bucket_by_sequence_length(
            processor.feats_length_fn,
            batch_conf['bucket_boundaries'],
            batch_conf['bucket_batch_sizes'],
            wrapper_class=partial(processor.padding,pass_cat_emb=pass_cat_emb, deep_biasing_conf=deep_biasing_conf))
            # wrapper_class=processor.padding)
    elif batch_type == 'distribute':
        logging.info("Starting distribute batch")
        max_frames_in_batch = batch_conf.get('max_frames_in_batch', 12000)
        one_utt_per_job = batch_conf.get('distrib_one_utt_per_job', True)
        max_words_per_epoch = batch_conf.get('distrib_max_word_count_per_epoch', -1)
        max_words_per_batch = batch_conf.get('distrib_max_word_count_per_batch', -1)
        verbose = batch_conf.get('verbose', False)
        #max_utt_from_single_job = batch_conf.get('max_utt_from_single_job', 2)
        dataset = dataset.distribute_batch(
            processor.DynamicBatchWindow(max_frames_in_batch),
            wrapper_class=partial(processor.padding,pass_cat_emb=pass_cat_emb, deep_biasing_conf=deep_biasing_conf),
            one_utt_per_job=one_utt_per_job,
            max_words_per_epoch=max_words_per_epoch,
            max_words_per_batch=max_words_per_batch,
            verbose=verbose
        )
    else:
        logging.info("Starting dynamic batch")
        max_frames_in_batch = batch_conf.get('max_frames_in_batch', 12000)
        dataset = dataset.dynamic_batch(
            processor.DynamicBatchWindow(max_frames_in_batch),
            wrapper_class=partial(processor.padding,pass_cat_emb=pass_cat_emb, deep_biasing_conf=deep_biasing_conf)
        )

    return dataset
