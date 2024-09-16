# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen)
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

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
import json
from torch.utils.data import DataLoader
from wenet.dataset.dataset import Dataset
import threading
import time
from typing import NamedTuple

from wenet.utils.executor import Executor
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import (
    # add_model_args,
    # init_distributed,
    # add_deepspeed_args,
    add_dataset_args, add_ddp_args,  init_dataset_and_dataloader )
import wenet.dataset.rev_processor as rev_processor
from wenet.utils.checkpoint import load_checkpoint

LossStatistics = NamedTuple("LossStatistics", [('dataset', str), ('checkpoint', str), ('loss', float), ('acc_att', float), ('time_to_process', float), ('loss_tel', float), ('acc_att_tel', float), ('loss_reverb', float), ('acc_att_reverb', float), ('loss_tel_reverb', float), ('acc_att_tel_reverb', float)]   )


import numpy as np
import psutil


def check_ram():
    global check_ram_loop
    check_ram_loop = True
    process = psutil.Process()
    while check_ram_loop:
        r = torch.cuda.memory_reserved(0)  / 1e9
        a = torch.cuda.memory_allocated(0) / 1e9
        open_files_count = len(process.open_files())
        print(f"{r:4.1f}GiB reserved, {a:4.1f}GiB allocated, {open_files_count} open files")
        time.sleep(0.5)

thread1 = threading.Thread(target=check_ram, args=())
# thread1.start()

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    # parser = add_model_args(parser)
    parser.add_argument('--train_engine',
                    default='torch_ddp',
                    choices=['torch_ddp', 'deepspeed'],
                    help='Engine for paralleled training')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=False, help='single checkpoint file')
    parser.add_argument('--enc_init', required=False, help='(ignore) initialize just the encoder')
    parser.add_argument('--override_config',
                    action='append',
                    default=[],
                    help="override yaml config")
    parser = add_dataset_args(parser)
    parser = add_ddp_args(parser)
    #parser = add_deepspeed_args(parser)
    #parser = add_trace_args(parser)
    # parser.add_argument('--ddp.dist_backend',
    #                 dest='dist_backend',
    #                 default='nccl',
    #                 choices=['nccl', 'gloo'],
    #                 help='distributed backend')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument("--jsonl_output", type=str, required=True, help="Output file with results in jsonl format (will be read and appended to if it already exists)")
    parser.add_argument('--checkpoints', nargs='+', help='checkpoints model')
    parser.add_argument('--extended',
                        action='store_true',
                        default=False,
                        help='Will run the data through telephony, reverb and telephony+reverb as well and report the 4 losses')
    args = parser.parse_args()
    return args

def get_statistics(jsonl_output):
    # the file is a list of json strings, one json document per line, with the following keys:
    # shard list, checkpoint path, cv_loss, cv_acc_att, time_to_process
    # We load that content into a list of LossStatistics namedtuples 
    if os.path.exists(jsonl_output):
        with open(jsonl_output, 'r') as f:
            lines = f.readlines()
        statistics = [LossStatistics(**json.loads(line)) for line in lines]
    else:
        statistics = []
    return statistics

def append_statistics(jsonl_output, statistics):
    with open(jsonl_output, 'a') as f:
        for stat in statistics:
            f.write(json.dumps(stat._asdict()) + '\n')

def dataset_and_checkpoint_exists(statistics, dataset, checkpoint):
    for stat in statistics:
        if stat.dataset == dataset and stat.checkpoint == checkpoint:
            return True
    return False


def disable__perturbations(configs):
    dsconf = configs['dataset_conf']
    dsconf['speed_perturb'] = False
    dsconf['spec_aug'] = False
    dsconf['spec_sub'] = False
    dsconf['spec_trim'] = False
    dsconf['apply_rir'] = False
    dsconf['apply_telephony'] = False
    dsconf['fbank_conf']['dither'] = False
    if dsconf['batch_conf'].get('batch_type', "static") == "distribute":
        dsconf['batch_conf']['batch_type'] = "dynamic"

    return configs


def get_plain_dataset(args, configs, tokenizer):
    generator = torch.Generator()
    generator.manual_seed(777)
    # let's make sure that the devset won't skip any utterances 
    dsconf = configs['dataset_conf']

    configs['vocab_size'] = tokenizer.vocab_size()
    dsconf = configs['dataset_conf']
    dataset = Dataset(args.data_type, args.cv_data, tokenizer, dsconf, partition=False)
    data_loader = DataLoader(dataset,
                            batch_size=None,
                            pin_memory=args.pin_memory,
                            num_workers=args.num_workers,
                            persistent_workers=False,
                            #persistent_workers=True,
                            generator=generator,
                            prefetch_factor=args.prefetch)
    return dataset, data_loader

def get_tel_config(real_configs, inplace=False):
    if inplace:
        configs = real_configs
    else:
        configs = copy.deepcopy(real_configs)
        cv_conf = configs['dataset_conf']
    cv_conf['apply_telephony'] = True
    if 'apply_telephony_conf' not in cv_conf:
        cv_conf['apply_telephony_conf'] = dict()
    cv_conf['apply_telephony_conf']['prob'] = 1
    return configs

def get_rir_config(real_configs, inplace=False):
    if inplace:
        configs = real_configs
    else:
        configs = copy.deepcopy(real_configs)
        cv_conf = configs['dataset_conf']
    cv_conf['apply_rir'] = True
    if 'apply_rir_conf' not in cv_conf:
        cv_conf['apply_rir_conf'] = dict()
        cv_conf['apply_rir_conf']['impulse_list_fn'] = '/shared/speech-db/VOiCES_devkit/impulse.list'

    cv_conf['apply_rir_conf']['prob'] = 1
    return configs

def doit():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.checkpoints is None or len(args.checkpoints) == 0:
       raise "For this script, the --checkpoint <model>.pt parameter is mandatory"

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # torch.set_default_device('cuda:' + args.gpu)
    # Set random seed
    torch.manual_seed(777)
    print(args)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # We're doing this so that we can take the training dataset handling to allow for augmentation to 
    # affect the CV loss computation
    args.train_data = args.cv_data
    world_size = 1
    distributed = False

    configs = disable__perturbations(configs)
    tokenizer = init_tokenizer(configs)

    cv_dataset, cv_data_loader = get_plain_dataset(args, configs, tokenizer)
    if args.extended:
        cv_dataset_tel_cfg = get_tel_config(configs)
        cv_dataset_rir_cfg = get_rir_config(configs)
        cv_dataset_tel_rir_cfg = get_rir_config(get_tel_config(configs))

        cv_dataset_tel, cv_data_loader_tel = get_plain_dataset(args, cv_dataset_tel_cfg, tokenizer)
        cv_dataset_rir, cv_data_loader_rir = get_plain_dataset(args, cv_dataset_rir_cfg, tokenizer)
        cv_dataset_tel_rir, cv_data_loader_tel_rir = get_plain_dataset(args, cv_dataset_tel_rir_cfg, tokenizer)

    # import sys
    # sys.exit(0)
    statistics = get_statistics(args.jsonl_output)
    print(f"We have {len(statistics)} statistics already in {args.jsonl_output}")

    # Uncomment the following if you want to trace GPU RAM usage over time
    # thread1.start()
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    saved_messages = []
    # we load the model once, we'll load checkpoints on the fly
    model, configs = init_model(args, configs)
    for chkfn in args.checkpoints:
        # infos = load_checkpoint(model, chkfn, force_cpu=True)
        # args.checkpoint = chkfn

        dataset_name = os.path.basename(args.cv_data)
        # check if the combination of dataset and checkpoint has already been processed
        if dataset_and_checkpoint_exists(statistics, dataset_name, chkfn):
            print(f"Skipping {chkfn} for dataset {args.cv_data}")
            continue
        
        infos = load_checkpoint(model, chkfn, def_strict=False)
        model.eval()
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device(f'cuda' if use_cuda else 'cpu')
        model.to(device)
        #wrap_cuda_model(args, model)
    
        final_epoch = None
        # configs['rank'] = args.rank
        configs['is_distributed'] = distributed
        # configs['use_amp'] = args.use_amp
        if True:
            t0 = time.time()
            loss_dict = executor.cv(model, cv_data_loader, configs)
            t1 = time.time()
            cv_loss = loss_dict['loss']
            acc_att = loss_dict['acc']
            time_to_process_in_secs = t1 - t0
        else:
            cv_loss = 0.0
            acc_att = 0.0
            time_to_process_in_secs = 0.0

        msg = f"{dataset_name} PLAIN : cv_loss {cv_loss:9.4f} cv_acc_att {acc_att:6.4f} for {chkfn} in {time_to_process_in_secs:.2f}s"
        if args.extended:
            t0 = time.time()
            loss_dict = executor.cv(model, cv_data_loader_tel, cv_dataset_tel_cfg)
            t1 = time.time()

            cv_loss_tel = loss_dict['loss']
            acc_att_tel = loss_dict['acc']
            time_to_process_in_secs_tel = t1 - t0

            t0 = time.time()
            loss_dict = executor.cv(model, cv_data_loader_rir, cv_dataset_rir_cfg)
            t1 = time.time()
            
            cv_loss_rir = loss_dict['loss']
            acc_att_rir = loss_dict['acc']
            time_to_process_in_secs_rir = t1 - t0

            t0 = time.time()
            loss_dict = executor.cv(model, cv_data_loader_tel_rir, cv_dataset_tel_rir_cfg)
            t1 = time.time()

            cv_loss_tel_rir = loss_dict['loss']
            acc_att_tel_rir = loss_dict['acc']
            time_to_process_in_secs_tel_rir = t1 - t0

            stat = LossStatistics(dataset=dataset_name, checkpoint=chkfn, loss=cv_loss, acc_att=acc_att, time_to_process=time_to_process_in_secs, loss_tel=cv_loss_tel, acc_att_tel=acc_att_tel, loss_reverb=cv_loss_rir, acc_att_reverb=acc_att_rir, loss_tel_reverb=cv_loss_tel_rir, acc_att_tel_reverb=acc_att_tel_rir)
            msg += f", TEL: loss {cv_loss_tel:9.4f} acc_att {acc_att_tel:6.4f} in {time_to_process_in_secs_tel:.2f}s"
            msg += f", RIR: loss {cv_loss_rir:9.4f} acc_att {acc_att_rir:6.4f} in {time_to_process_in_secs_rir:.2f}s"
            msg += f", TEL RIR: loss {cv_loss_tel_rir:9.4f} acc_att {acc_att_tel_rir:6.4f}  {time_to_process_in_secs_tel_rir:.2f}s"
        else:
            # create the LossStatics object
            stat = LossStatistics(dataset=dataset_name, checkpoint=chkfn, loss=cv_loss, acc_att=acc_att, time_to_process=time_to_process_in_secs, loss_tel=None, acc_att_tel=None, loss_reverb=None, acc_att_reverb=None, loss_tel_reverb=None, acc_att_tel_reverb=None)
        print(msg)

        statistics.append(stat)
        append_statistics(args.jsonl_output, [stat]) 

        saved_messages.append(msg)


    print("FINAL recap:")
    for msg in saved_messages:
        print(msg)
    print("done")


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(1)
    try:
        doit()
    except KeyboardInterrupt:
        print("ctrl-c")
    finally:
        check_ram_loop = False
        #thread1.join()
