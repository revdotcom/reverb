# Copyright (c) 2023 Wenet Community. (authors: Dinghao Zhou)
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

import collections
from collections.abc import Callable
import sys
import tarfile
import zipfile
import logging
from typing import List
import torch
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data import datapipes
from torch.utils.data.datapipes.iter import Mapper
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES, ShardingFilterIterDataPipe)
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

from torch.profiler import profile, record_function, ProfilerActivity
import traceback
from wenet.dataset.processor import parse_url
import gc
from torch.utils.data import  get_worker_info


from multiprocessing import Manager
from multiprocessing.shared_memory import ShareableList
import threading
import os
# manager = Manager()

#def apply_random_seed_overwrite(datapipe: torch.utils.data.graph.DataPipe, rng: torch.Generator) -> torch.utils.data.graph.DataPipe:
#        return datapipe
#
#
#torch.utils.data.graph_settings.apply_random_seed = apply_random_seed_overwrite

@functional_datapipe("map_ignore_error")
class MapperIgnoreErrorDataPipe(Mapper):

    def __init__(self,
                 dataset: IterDataPipe,
                 fn: Callable,
                 input_col=None,
                 output_col=None,
                 log_error: bool = True) -> None:
        super().__init__(dataset, fn, input_col, output_col)
        self._iter = None
        self.log_error = log_error

    def __iter__(self):
        if self._iter is None:
            self._iter = iter(self.datapipe)

        while True:
            try:
                elem = next(self._iter)
                yield self._apply_fn(elem)
            except StopIteration:
                self._iter = None
                return
            except Exception as ex:
                if True or self.log_error:
                    logging.warning(f"xx : {ex}, {repr(ex)}")
                    traceback.print_exc()


@functional_datapipe('bucket_by_sequence_length')
class BucketBySequenceLengthDataPipe(IterDataPipe):

    def __init__(
        self,
        dataset: IterDataPipe,
        elem_length_func,
        bucket_boundaries: List[int],
        bucket_batch_sizes: List[int],
        wrapper_class=None,
    ) -> None:
        super().__init__()
        _check_unpickable_fn(elem_length_func)
        assert len(bucket_batch_sizes) == len(bucket_boundaries) + 1
        self.bucket_batch_sizes = bucket_batch_sizes
        self.bucket_boundaries = bucket_boundaries + [sys.maxsize]
        self.elem_length_func = elem_length_func

        self._group_dp = GroupByWindowDataPipe(dataset,
                                               self._element_to_bucket_id,
                                               self._window_size_func,
                                               wrapper_class=wrapper_class)

    def __iter__(self):
        yield from self._group_dp

    def _element_to_bucket_id(self, elem):
        seq_len = self.elem_length_func(elem)
        bucket_id = 0
        for (i, b) in enumerate(self.bucket_boundaries):
            if seq_len < b:
                bucket_id = i
                break
        return bucket_id

    def _window_size_func(self, bucket_id):
        return self.bucket_batch_sizes[bucket_id]


@functional_datapipe("group_by_window")
class GroupByWindowDataPipe(datapipes.iter.Grouper):

    def __init__(
        self,
        dataset: IterDataPipe,
        key_func,
        window_size_func,
        wrapper_class=None,
    ):
        super().__init__(dataset,
                         key_func,
                         keep_key=False,
                         group_size=None,
                         drop_remaining=False)
        _check_unpickable_fn(window_size_func)
        self.dp = dataset
        self.window_size_func = window_size_func
        if wrapper_class is not None:
            _check_unpickable_fn(wrapper_class)
            del self.wrapper_class
            self.wrapper_class = wrapper_class

    def __iter__(self):
        for x in self.datapipe:
            key = self.group_key_fn(x)

            self.buffer_elements[key].append(x)
            self.curr_buffer_size += 1

            group_size = self.window_size_func(key)
            if group_size == len(self.buffer_elements[key]):
                result = self.wrapper_class(self.buffer_elements[key])
                yield result
                self.curr_buffer_size -= len(self.buffer_elements[key])
                del self.buffer_elements[key]

            if self.curr_buffer_size == self.max_buffer_size:
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    result = self.wrapper_class(result_to_yield)
                    yield result

        for key in tuple(self.buffer_elements.keys()):
            result = self.wrapper_class(self.buffer_elements.pop(key))
            self.curr_buffer_size -= len(result)
            yield result


@functional_datapipe("sort")
class SortDataPipe(IterDataPipe):

    def __init__(self,
                 dataset: IterDataPipe,
                 buffer_size: int = 500,
                 key_func=None,
                 reverse=False) -> None:
        if key_func is not None:
            _check_unpickable_fn(key_func)
        self.buffer_size = buffer_size
        super().__init__()
        self.dp = dataset
        # self.manager = Manager()
        # self._buffer = self.manager.list([])
        self._buffer = []
        # self._buffer = ShareableList([])
        self.key_func = key_func
        self.reverse = reverse

    def __iter__(self):
        for elem in self.dp:
            self._buffer.append(elem)
            if len(self._buffer) >= self.buffer_size:
                self._buffer.sort(key=self.key_func, reverse=self.reverse)
                for x in self._buffer:
                    yield x
                del self._buffer
                # self._buffer = self.manager.list([])
                # self._buffer = ShareableList([])
                self._buffer = []
        # The sample left over
        self._buffer.sort(key=self.key_func, reverse=self.reverse)
        for x in self._buffer:
            yield x
        del self._buffer
        # self._buffer = self.manager.list([])
        # self._buffer = ShareableList([])
        self._buffer = []

@functional_datapipe("distribute_batch")
class DistributeBatchDataPipe(IterDataPipe):
    count_instances = 0

    def __init__(self, dataset: IterDataPipe, window_class,
                 wrapper_class,
                 one_utt_per_job=True,
                 max_words_per_epoch=-1,
                 max_words_per_batch=-1,
                 verbose=False,
                 ) -> None:
        print("initializing a DistributeBatchDataPipe")
        _check_unpickable_fn(window_class)
        _check_unpickable_fn(wrapper_class)
        super().__init__()
        self.dp = dataset
        assert window_class is not None
        assert wrapper_class is not None
        self.window_class = window_class
        # self.manager = Manager()
        # self._buffer = self.manager.list([])
        # self._buffer = ShareableList([])
        self._buffer = []
        self._wrappr_class = wrapper_class
        self._one_utt_per_job = one_utt_per_job
        self._max_words_per_epoch=max_words_per_epoch
        self._max_words_per_batch=max_words_per_batch
        self.verbose = verbose
        #if True:
        with threading.Lock(): 
            DistributeBatchDataPipe.count_instances += 1
            self.rand_id = f"{os.getpid()}-{threading.current_thread().name}-{DistributeBatchDataPipe.count_instances}-{id(self)}"
        logging.info(f"new DistributeBatchDataPipe with rand_id {self.rand_id} started ")


    @staticmethod
    def _get_job_id(key):
        parts = key.split(".")
        if len(parts) >= 3:
            return parts[0] + "." + parts[1]

    def __iter__(self):
        self._seen_words = dict()
        self._send_utt = set()
        reported_elements = []
        drop_utts = 0
        drop_frames = 0
        num_batches = 0

        batch_words = {}
        epoch_words = {}

        worker_info = get_worker_info()

        #while True:
        for elem in self.dp:
            utt = elem['key']
            txt = elem['txt']
            wds = txt.split(" ")

            # using 2 here will preserve the speaker id as part of the key.
            # TC0000000P-00.aligned.nlp.wav_speaker00001-TC000000-00.m4a-A-000
            # TC000000P-1.aligned.nlp.wav_speaker00000_00001-TC000000P-1.m4a-A-000
            # TODO: make configurable, include speaker or not
            #split_utt = utt.split(".", 2)
            #rev_job_id = ".".join(split_utt[:2])
            rev_job_id = DistributeBatchDataPipe._get_job_id(utt)
            if self._one_utt_per_job and rev_job_id in self._send_utt:
                drop_utts += 1
                drop_frames += elem['feat'].size(0)
                del elem
                continue

            if self._max_words_per_epoch > 0:
                accept = any(epoch_words.get(s, 0) < self._max_words_per_epoch for s in wds)
                if not accept:
                    #print(f"rejected {txt}")
                    drop_utts += 1
                    drop_frames += elem['feat'].size(0)
                    del elem
                    continue

            if self._max_words_per_batch > 0:
                accept = any(batch_words.get(s, 0) < self._max_words_per_batch for s in wds)
                if not accept:
                    drop_utts += 1
                    drop_frames += elem['feat'].size(0)
                    del elem
                    continue

            if  self._one_utt_per_job:
                self._send_utt.add(rev_job_id)

            if self._max_words_per_epoch > 0:
                for s in wds:
                    epoch_words[s] = epoch_words.get(s, 0) + 1

            if self._max_words_per_batch > 0:
                for s in wds:
                    batch_words[s] = batch_words.get(s, 0) + 1
                    #print(f"{s=}, batch_words[s] = {batch_words[s]}")

            #print(f"{self._max_words_per_batch=}, {self._max_words_per_epoch}")
            #print(wds)
            #print(batch_words)
            if not self.window_class(elem, len(self._buffer)):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    num_batches += 1
                    if self.verbose and num_batches % 100 == 0:
                        logging.info(f"after {num_batches} batches : We dropped {drop_utts} utts and {drop_frames} frames to build this batch, rand_id {self.rand_id}, th {worker_info.id}")
                    yield self._wrappr_class(self._buffer)
                del self._buffer
                batch_words = {}
                self._send_utt = set()
                #print(f"We dropped {drop_utts} utts and {drop_frames} frames to build this batch")
                #drop_utts =0
                #drop_frames =0
                self._buffer = []
        if len(self._buffer) > 0:
            yield self._wrappr_class(self._buffer)
        logging.info(f"final : We dropped {drop_utts} utts and {drop_frames} frames to build this batch")
        epoch_words = {}
        del self._buffer
        self._buffer = []

@functional_datapipe("dynamic_batch")
class DynamicBatchDataPipe(IterDataPipe):

    def __init__(self, dataset: IterDataPipe, window_class,
                 wrapper_class) -> None:
        _check_unpickable_fn(window_class)
        _check_unpickable_fn(wrapper_class)
        super().__init__()
        self.dp = dataset
        assert window_class is not None
        assert wrapper_class is not None
        self.window_class = window_class
        # self.manager = Manager()
        # self._buffer = self.manager.list([])
        # self._buffer = ShareableList([])
        self._buffer = []
        self._wrappr_class = wrapper_class

    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, len(self._buffer)):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    yield self._wrappr_class(self._buffer)
                del self._buffer
                # self._buffer = self.manager.list([elem])
                # self._buffer = ShareableList([])
                self._buffer = []
        if len(self._buffer) > 0:
            yield self._wrappr_class(self._buffer)
        del self._buffer
        # self._buffer = self.manager.list([])
        # self._buffer = ShareableList([])
        self._buffer = []


@functional_datapipe("prefetch")
class PrefetchDataPipe(IterDataPipe):
    """Performs prefetching"""

    def __init__(
        self,
        dataset: IterDataPipe,
        buffer_size: int = 500,
    ):
        # TODO(Mddct): support multiprocessing pool with shared-memory to
        #   prefetch
        super().__init__()
        self.dp = dataset
        self._iter = None
        self._prefetch_buffer_size = buffer_size
        self._buffer = None
        if self._prefetch_buffer_size > 0:
            self._buffer = collections.deque(maxlen=self._prefetch_buffer_size)

    def __iter__(self):
        if self._prefetch_buffer_size > 0:
            if self._iter is None:
                self._iter = iter(self.dp)
            assert self._buffer is not None

            while True:
                if len(self._buffer) <= self._prefetch_buffer_size // 2:
                    while len(self._buffer) < self._prefetch_buffer_size:
                        try:
                            self._buffer.append(next(self._iter))
                        except StopIteration:
                            if len(self._buffer) != 0:
                                while len(self._buffer) > 0:
                                    yield self._buffer.popleft()
                            self._iter = None
                            return
                while len(self._buffer) > self._prefetch_buffer_size // 2:
                    elem = self._buffer.popleft()
                    yield elem

        else:
            yield from self.dp


@functional_datapipe("shard")
class ShardDataPipe(ShardingFilterIterDataPipe):

    def __init__(self, dataset: IterDataPipe, partition: bool = False):
        super().__init__(dataset, None)
        self.partition = partition
        self.dp = dataset

    def apply_sharding(self, num_of_instances: int, instance_id: int,
                       sharding_group: SHARDING_PRIORITIES):
        if self.partition:
            return super().apply_sharding(num_of_instances, instance_id,
                                          sharding_group)
        else:
            # We can not handle uneven data for CV on DDP, so we don't
            # sample data by rank, that means every GPU gets the same
            # and all the CV data
            info = torch.utils.data.get_worker_info()
            if info is None:
                self.num_of_instances = 1
                self.instance_id = 0
            else:
                n_workers_per_device = info.num_workers
                self.num_of_instances = n_workers_per_device
                self.instance_id = info.id


class TextLineDataPipe(IterDataPipe):
    """ Streamming Text line
    """

    def __init__(self, filenames, mode='r'):
        super().__init__()
        self.mode = mode
        _dp = datapipes.iter.FileLister(filenames)
        _dp = datapipes.iter.FileOpener(_dp, mode=mode)
        #_dp = [c for c in datapipes.iter.FileOpener(_dp, mode=mode)]
        # self.manager = Manager()
        self.dp = _dp

    def __iter__(self):
        #mode = self.mode
        #_dp = datapipes.iter.FileOpener(self.dp, mode=mode)
        for fname, stream in self.dp:
        #for fname, stream in _dp:
            for line in stream:
                line = line.strip('\n')
                elm = {"file_name": fname, "line": line}
                yield elm
                del elm
            stream.close()

@functional_datapipe("single_dict_wrapper")
class ElementWrapperDataPipe(IterDataPipe):
    """ Yield a single example
    """
    def __init__(self, elements : dict) -> None:
        super().__init__()
        self.elements = elements

    def __iter__(self):
        yield self.elements

@functional_datapipe("unpack_file_and_group")
class ArchiveTypeSelectorDataPipe(IterDataPipe):
    """ Decode any archive, zip or tar, using the appropriate underlying class
    """
    def __init__(self, dataset: IterDataPipe) -> None:
        super().__init__()
        self.dp = dataset

    def __iter__(self):
        # print(f"ArchiveTypeSelector, type of dp = {type(self.dp)}")
        if True:
            try:
                for sample in self.dp:
                    # print(f"ArchiveTypeSelector XXXXX, type:({type(sample)}), {sample}")
                    assert 'file_name' in sample  # train.shards.list
                    assert 'line' in sample       # /path/to/archive.0000.zip
                    assert 'stream' in sample     # above stream open
                    # gc.collect()

                    if True:
                        if sample['line'].endswith("tar.gz"):
                            # print(f"ArchiveTypeSelector: tar file {sample['line']}")
                            try:
                                for elm in TarsDataPipeJp.handle_tar_file(sample):
                                    yield elm
                                    del elm
                            except Exception as ex:
                                msg = 'TarsDataPipeline In tar_file_and_group: {} when processing {}'.format(
                                    ex, sample['line'])
                                logging.warning(msg)
                            finally:
                                if 'process' in sample:
                                    sample['process'].communicate()
                                sample['stream'].close()
                        elif sample['line'].endswith(".zip"):
                            # print(f"ArchiveTypeSelector: zip file {sample['line']}")
                            #with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
                            if True:
                                try:
                                    for elm in ZipsDataPipe.handle_zip_file(sample):
                                        yield elm
                                        del elm
                                except Exception as err:
                                    logging.warning('(1) issue with {}'.format(sample['src']))
                                    print(Exception, err)
                                finally:
                                    try:
                                        if 'process' in sample:
                                            #print("closing process()")
                                            sample['process'].communicate()
                                        #print("closing stream")
                                        sample['stream'].close()
                                    except Exception as err:
                                        logging.warning('(2) issue with {}'.format(sample['src']))
                                        print(Exception, err)
                                        pass
                            #print(f"For {sample['line']}: {prof.key_averages().table(sort_by='self_cpu_memory_usage', row_limit=20)}")
                        
            except Exception as ex2:
                print(f"In ArchiveSelector iter, exception thrown {ex2} : {traceback.format_exc()}")
                # yield from ZipsDataPipe(ElementWrapperDataPipe(sample))

@functional_datapipe("zip_file_and_group")
class ZipsDataPipe(IterDataPipe):
    """ Decode wenet's zip , yield {'txt': "...", "raw": "..."}
    """
    def __init__(self, dataset: IterDataPipe) -> None:
        super().__init__()
        self.dp = dataset

    @staticmethod
    def handle_zip_file(sample):
        num_valid_example_in_zip = 0
        num_total_example_in_zip = 0
        with zipfile.ZipFile(sample['stream']) as stream:
            prev_prefix = None
            # Manager().dict(example = {
            #     'file_name': sample['file_name'],
            #     'tar_file_name': sample['line']
            # })
            example = {
                'file_name': sample['file_name'],
                'tar_file_name': sample['line']
            }
            valid = True
            wav_list = [file_name for file_name in stream.namelist() if file_name.rsplit('.', maxsplit=1)[1] == 'wav']
            postfixes = set((file_name.rsplit(
                '.', maxsplit=1)[-1] for file_name in stream.namelist())) - set(['wav'])
            for wav_file in wav_list:
                prefix, postfix = wav_file.rsplit('.', maxsplit=1)
                example['key'] = prefix
                num_total_example_in_zip += 1
                # mystats['in_zip'] += 1
                try:
                    # waveform, sample_rate = torchaudio.load( stream.open(wav_file))
                    # example['wav'] = waveform
                    # example['sample_rate'] = sample_rate
                    # example['wav'] = tensor.tensor(stream.open(wav_file).read())
                    example['wav'] = stream.open(wav_file).read()
                    for postfix in postfixes:
                        try:
                            example[postfix] = stream.read(
                                prefix + '.' + postfix).decode('utf8').strip()
                        except:
                            # JPR : let's do this a better eventually...
                            pass
                    if valid:
                        num_valid_example_in_zip += 1
                        yield example
                    # example = Manager().dict({})
                    example = {}
                    valid = True
                except Exception as ex:
                    valid = False
                    logging.warning(
                        'in zip file, error to parse {}, postfix = [{}]'.format(prefix, postfix))
                    logging.warning(ex)
            #try:
            #    #stream.close()
            #    if 'process' in sample:
            #        sample['process'].communicate()
            #    sample['stream'].close()
            #except Exception as err:
            #    logging.warning('(2) issue with {}'.format(sample['src']))
            #    print(Exception, err)
            #    pass

    def __iter__(self):
        from wenet.dataset.processor import AUDIO_FORMAT_SETS
        for sample in self.dp:
            assert 'file_name' in sample
            assert 'line' in sample
            assert 'stream' in sample
            try:
                yield ZipsDataPipe.handle_zip_file(sample)
            except Exception as err:
                logging.warning('(1) issue with {}'.format(sample['src']))
                print(Exception, err)
            finally:
                try:
                    if 'process' in sample:
                        sample['process'].communicate()
                    sample['stream'].close()
                except Exception as err:
                    logging.warning('(2) issue with {}'.format(sample['src']))
                    print(Exception, err)
                    pass


@functional_datapipe("tar_file_and_group2")
class TarsDataPipeJp(IterDataPipe):
    """ Decode wenet's tar , yield {'txt': "...", "raw": "..."}
    """

    def __init__(self, dataset: IterDataPipe) -> None:
        super().__init__()
        self.dp = dataset

    @staticmethod
    def handle_tar_file(sample):
        from wenet.dataset.processor import AUDIO_FORMAT_SETS
        assert 'file_name' in sample
        assert 'line' in sample
        assert 'stream' in sample
        with tarfile.open(fileobj=sample['stream'],
                            mode="r:*") as stream:
            prev_prefix = None
            example = {
                'file_name': sample['file_name'],
                'tar_file_name': sample['line']
            }
            valid = True
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    example['key'] = prev_prefix
                    if valid:
                        yield example
                    example = {
                        'file_name': sample['file_name'],
                        'tar_file_name': sample['line']
                    }
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if postfix == 'txt':
                            example['txt'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix in AUDIO_FORMAT_SETS:
                            example['wav'] = file_obj.read()
                        else:
                            example[postfix] = file_obj.read().decode('utf8').strip()
                    except Exception as ex:
                        valid = False
                        logging.warning(
                            'error to parse {}, {}'.format(name, ex))
                    prev_prefix = prefix
            if prev_prefix is not None:
                example['key'] = prev_prefix
                yield example

    def __iter__(self):
        from wenet.dataset.processor import AUDIO_FORMAT_SETS
        for sample in self.dp:
            assert 'file_name' in sample
            assert 'line' in sample
            assert 'stream' in sample
            try:
                for elm in TarsDataPipe.handle_tar_file(sample):
                    yield elm
            except Exception as ex:
                msg = 'TarsDataPipeline In tar_file_and_group2: {} when processing {}'.format(
                    ex, sample['line'])
                logging.warning(msg)
            finally:
                if 'process' in sample:
                    sample['process'].communicate()
                sample['stream'].close()

@functional_datapipe("tar_file_and_group")
class TarsDataPipeWenet(IterDataPipe):
    """ Decode wenet's tar , yield {'txt': "...", "raw": "..."}
    """

    def __init__(self, dataset: IterDataPipe) -> None:
        super().__init__()
        self.dp = dataset

    def __iter__(self):
        from wenet.dataset.processor import AUDIO_FORMAT_SETS
        for sample in self.dp:
            assert 'file_name' in sample
            assert 'line' in sample
            assert 'stream' in sample
            try:
                with tarfile.open(fileobj=sample['stream'],
                                  mode="r:*") as stream:
                    prev_prefix = None
                    example = {
                        'file_name': sample['file_name'],
                        'tar_file_name': sample['line']
                    }
                    valid = True
                    for tarinfo in stream:
                        name = tarinfo.name
                        pos = name.rfind('.')
                        assert pos > 0
                        prefix, postfix = name[:pos], name[pos + 1:]
                        if prev_prefix is not None and prefix != prev_prefix:
                            example['key'] = prev_prefix
                            if valid:
                                yield example
                            example = {
                                'file_name': sample['file_name'],
                                'tar_file_name': sample['line']
                            }
                            valid = True
                        with stream.extractfile(tarinfo) as file_obj:
                            try:
                                if postfix == 'txt':
                                    example['txt'] = file_obj.read().decode(
                                        'utf8').strip()
                                elif postfix in AUDIO_FORMAT_SETS:
                                    example['wav'] = file_obj.read()
                                else:
                                    example[postfix] = file_obj.read().decode('utf8').strip()
                            except Exception as ex:
                                valid = False
                                logging.warning(
                                    'error to parse {}'.format(name))
                            prev_prefix = prefix
                    if prev_prefix is not None:
                        example['key'] = prev_prefix
                        yield example
            except Exception as ex:
                msg = 'In tar_file_and_group: {} when processing {}'.format(
                    ex, sample['line'])
                logging.warning(msg)
            finally:
                if 'process' in sample:
                    sample['process'].communicate()
                sample['stream'].close()




class WenetRawDatasetSource(IterDataPipe):

    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 partition=True) -> None:
        super().__init__()
        #self.dp = TextLineDataPipe(filenames).prefetch(prefetch).shard(partition)
        self.dp = TextLineDataPipe(filenames).prefetch(prefetch).shard(partition)

    def __iter__(self):
        for d in self.dp:
            yield d


#class WenetTarShardDatasetSource(IterDataPipe):
#
#    def __init__(self,
#                 filenames: str,
#                 prefetch: int = 500,
#                 partition: bool = False) -> None:
#        super().__init__()
#        self.dp = TextLineDataPipe(filenames).shard(
#            partition).map_ignore_error(
#                # parse_url).tar_file_and_group().prefetch(prefetch)
#                parse_url).unpack_file_and_group().prefetch(prefetch)
#
#    def __iter__(self):
#        for d in self.dp:
#            yield d
class WenetTarShardDatasetSource(IterDataPipe):
    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 partition: bool = True,
                 shuffle: bool = False,
                 shuffle_size: int = 10000,
                 cycle: int = 1) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames)
        if shuffle:
            self.dp = self.dp.shuffle(buffer_size=shuffle_size)
        #self.dp = self.dp.repeat(cycle)
        self.dp = self.dp.shard(partition).map_ignore_error(
            parse_url).unpack_file_and_group().prefetch(prefetch)
            #parse_url).tar_file_and_group().prefetch(prefetch)

    def __iter__(self):
        for d in self.dp:
            yield d
