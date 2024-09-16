# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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
import shutil
import sys

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import wandb

from wenet.dataset.dataset import Dataset
from wenet.utils.init_model import init_model
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint
from wenet.utils.executor import Executor
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.scheduler import WarmupLR
import torch_optimizer as optim_ex
from wenet.utils.config import override_config
import wenet.dataset.processor as processor
#import pdb_attach

# These two lines won't help much because our input sizes are changing constantly
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
def get_peak_memory_MB(device):
    return torch.cuda.max_memory_allocated(device) // 1e6

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--exclude_keys', required=False, help='file with utterance keys to exclude')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--freeze_encoder',
                        action='store_true',
                        default=False,
                        help='Freeze encoder weights')
    parser.add_argument('--restore_encoder_only',
                        action='store_true',
                        default=False,
                        help='Restore only encoders weights from checkpoint')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--snap_every',
                        default=3000,
                        type=int,
                        help='create snapshot after every X batches')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument('--gather_max_memory_stats',
                        action='store_true',
                        default=False,
                        help='At the start of training pass a few maximum '
                             'sized tensors to probe for memory failures '
                             'and gather statistics.')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')


    args.gpu        = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank       = int(os.environ["RANK"])

    # maybe we need to remove this for torchrun?
    # and change the .cuda() to .to(device) ?
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Set random seed
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    
    distributed = args.world_size > 1
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        # simplified by torchrun
        dist.init_process_group(args.dist_backend)
        # dist.init_process_group(args.dist_backend,
        #                         init_method=args.init_method,
        #                         world_size=args.world_size,
        #                         rank=args.rank)

    symbol_table = read_symbol_table(args.symbol_table)

    # port = 50000 + args.rank
    # logging.info(f"pdb attached for process {os.getpid()} to port {port} for rank {args.rank}")
    # pdb_attach.listen(port)
    configs['snap_every'] = args.snap_every

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['apply_rir'] = False
    cv_conf['apply_telephony'] = False

    train_dataset = Dataset(args.data_type, args.train_data, symbol_table,
                            train_conf, args.bpe_model, partition=True, exclude_keys=args.exclude_keys)

    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         symbol_table,
                         cv_conf,
                         args.bpe_model,
                         partition=False)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)

    input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    vocab_size = len(symbol_table)

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = configs.get('is_json_cmvn', True)
    if args.rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init asr model from configs
    model = init_model(configs)
    if args.rank == 0:
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        print('the number of model params: {}'.format(num_params))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    if args.rank == 0:
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    if args.rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        if "WANDB_PROJECT" in os.environ and os.environ["WANDB_PROJECT"] != "":
            wandb_project_id=os.environ["WANDB_PROJECT"]
        else:
            wandb_project_id=exp_id

        WANDB_KEY = os.environ["WANDB_KEY"]
        WANDB_HOST = os.environ["WANDB_HOST"]
        wandb.login(host=WANDB_HOST, key=WANDB_KEY)

        wandb.init(
            project=wandb_project_id,
            config=configs,
        )

        artifact = wandb.Artifact('wenet-tree', type='code', description="snapshot of the wenet folders at launch time")
        artifact.add_dir("wenet/")
        wandb.log_artifact(artifact)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        use_cuda = True
        device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
        model.to(device)
        logging.info(f"DEVICE is {device}")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
        device = torch.device("cuda")
        if args.rank == 0:
            wandb.watch(model, log_freq=5000)
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    if configs['optim'] == "novograd":
        print("preparing novograd optimizer")
        # optimizer = optim_ex.NovoGrad(model.parameters(), **configs['optim_conf'])
        optimizer = optim_ex.NovoGrad(model.parameters(), lr=configs['optim_conf']['lr'], betas=(0.8, 0.25))
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    elif configs['optim'] == "radam":
        print("preparing radam optimizer")
        optimizer = optim.RAdam(model.parameters(), **configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    elif configs['optim'] == "zero_adam":
        print("preparing ZeroRedundancy Adam optimizer")
        optimizer = ZeroRedundancyOptimizer(
                model.parameters(),
                optimizer_class=torch.optim.Adam,
                **configs['optim_conf'] 
                )
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    else:
        print("preparing adam optimizer")
        optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])

    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        if distributed:
            infos = load_checkpoint(model.module, args.checkpoint, args.restore_encoder_only, force_cpu=True, optimizer=optimizer)
        else:
            infos = load_checkpoint(model, args.checkpoint, args.restore_encoder_only, force_cpu=True, optimizer=optimizer)
    else:
        infos = {}

    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    final_epoch = None
    configs['rank'] = args.rank
    configs['is_distributed'] = distributed
    configs['use_amp'] = args.use_amp
    if start_epoch == 0 and args.rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path, optimizer=optimizer)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # let's try to do one pass into the model with the max-size tensors to prepare the 
    # way for memory usage
    if args.gather_max_memory_stats:
        print("memory stats before for {device}")
        print(torch.cuda.memory_stats(device=device))
        b_max_len = configs['dataset_conf']['filter_conf']['max_length']
        b_max_tk_len = configs['dataset_conf']['filter_conf']['token_max_length']
        b_bs = configs['dataset_conf']['batch_conf']['batch_size']

        xs = torch.rand(b_bs,b_max_len - 2,  80, device=device)
        xs_len = torch.ones(b_bs, dtype=torch.int, device=device) * b_max_len

        ys = torch.randint(3, 10000, (b_bs, b_max_tk_len-2), device=device)
        ys_len = torch.ones(b_bs, dtype=torch.int, device=device) * b_max_tk_len
        
        should_raise = False
        try :
            model.train()
            for xx in range(4):
                # loss, loss_att, loss_ctc, acc_att = model(xs, xs_len, ys, ys_len)
                losses_and_other = model(xs, xs_len, ys, ys_len)
                loss = losses_and_other["loss"]
                loss = loss / 4
                xs = torch.rand(b_bs,b_max_len,  80, device=device)
                xs_len = torch.ones(b_bs, dtype=torch.int, device=device) * b_max_len

                ys = torch.randint(3, 10000, (b_bs, b_max_tk_len), device=device)
                ys_len = torch.ones(b_bs, dtype=torch.int, device=device) * b_max_tk_len
            optimizer.zero_grad(set_to_none=True)
        except:
            should_raise = False
        finally:
            print("memory stats after for {device}")
            print(torch.cuda.memory_stats(device=device))
            print("memory summary after for {device}")
            print(torch.cuda.memory_summary(device=device))
        if should_raise :
            raise("there was an error")


    for epoch in range(start_epoch, num_epochs):
        logging.info(f"starting {epoch} of {num_epochs} epoch on rank {args.rank}")
        sys.stdout.flush()
        sys.stderr.flush()

        train_dataset.set_epoch(epoch)
        configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, configs, scaler, args.rank, args.world_size, cv_data_loader, model_dir, configs, epoch)

        
        logging.info(f"calling end of epoch CV for {epoch} epoch on rank {args.rank}")
        logging.info(f"rank {args.rank} Epoch {epoch} num steps = {executor.step}")
        sys.stdout.flush()
        sys.stderr.flush()
        with model.no_sync():
            total_loss, total_att_acc, cv_num_seen_utts = executor.cv(model, cv_data_loader, device, configs)

        cv_loss = total_loss / cv_num_seen_utts
        att_acc = total_att_acc / cv_num_seen_utts

        logging.info(f"DONE end of epoch CV for {epoch} epoch on rank {args.rank}")
        sys.stdout.flush()
        sys.stderr.flush()

        #cv_loss = total_loss / num_seen_utts
        #att_acc = total_att_acc / num_seen_utts

        logging.info('End of Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        sys.stdout.flush()
        sys.stderr.flush()
        if args.rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            logging.info('Saving model to {}'.format(save_model_path))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'att_acc' : att_acc.item(),
                    'step': executor.step
                }, 
                optimizer)
            writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
            wandb_log = {'cv_loss': cv_loss, 'lr': lr, 'epoch' : epoch, 'cv_acc_att': att_acc }
            wandb.log(wandb_log)
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        if os.path.exists(final_model_path):
            shutil.move(final_model_path, final_model_path + '.bak')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    try:
        main()
    finally:
        print("stats")
        print(processor.mystats)
