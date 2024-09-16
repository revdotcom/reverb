# Copyright (c) 2020 Mobvoi Inc (Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import argparse
import glob
from collections import defaultdict
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import numpy as np
import yaml
import torch
import wandb
import tqdm, time


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=False,
                        help='src model path for average - can use --wandb_project instead')
    parser.add_argument('--wandb_project',
                        required=False,
                        help='wandb project for average')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='use validation accuracy to rank snapshots')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')
    parser.add_argument('--min_epoch',
                        default=0,
                        type=int,
                        help='min epoch used for averaging model')
    parser.add_argument('--max_epoch',
                        default=-1,
                        type=int,
                        help='max epoch used for averaging model')
    parser.add_argument('--use_att_acc', # doesn't do anything 
                        action="store_true",
                        help='Use maximum attention accuracy instead of cv_loss to select best snapshots')
    parser.add_argument('--from_snapshot', 
                        action="store_true",
                        help='whether to use all training snapshots or only end-of-epoch snapshots')
    parser.add_argument('--run_tag',
                        default=None,
                        type=str,
                        help='filter snapshots by wandb artifact "run_tag" field')
    parser.add_argument('--run_name',
                        default=None,
                        type=str,
                        help='filter snapshots by wandb run name')
    parser.add_argument('--min_step', # doesn't do anything
                        default=0,
                        type=int,
                        help='min step used for averaging model')
    parser.add_argument('--max_step', # doesn't do anything
                        default=sys.maxsize,
                        type=int,
                        help='max step used for averaging model')

    args = parser.parse_args()
    if not (args.src_path or args.wandb_project):
        raise Exception("Must specify either src_path or wandb_project for model averaging.")
    print(args)
    return args



def get_snapshots_local(src_path, val_best, num, min_epoch, max_epoch, use_att_acc, from_snapshot, min_step, max_step):
    """ yields loaded pytorch models to be averaged

    Input:
      src_path -- path to a directory of model snapshots
      val_best -- if True, use validation loss to rank snapshots
      num -- number of models to average
      min_epoch -- minimum epoch of snapshots to select
      max_epoch -- maximum epoch of snapshots to select 
      use_att_acc -- if True, use attention accuracy on the validation set to rank snapshots
      from_snapshot -- if True, can include mid-epoch snapshots in average instead of only using end-of-epoch models
      min_step -- minimum training step of snapshots to select
      max_step -- maximum training step of snapshots to select
    """
    if args.val_best:
        if args.from_snapshot:
            yamls = glob.glob('{}/snapshot_*.yaml'.format(args.src_path))
        else:
            yamls = list(set(glob.glob('{}/*.yaml'.format(args.src_path)))
                         - (
                         set(glob.glob('{}/init.yaml'.format(args.src_path)))
                         .union(set(glob.glob('{}/train.yaml'.format(args.src_path)))
                         .union(set(glob.glob('{}/snapshot*.yaml'.format(args.src_path)))))
                         ))
            #yamls = glob.glob('{}/[!train][!snapshot]*.yaml'.format(args.src_path))

        print(f"yamls : {yamls}")
        att_acc = -1.0
        if args.mode == "hybrid":
            yamls = glob.glob('{}/*.yaml'.format(args.src_path))
            yamls = [
                f for f in yamls
                if not (os.path.basename(f).startswith('train')
                        or os.path.basename(f).startswith('init'))
            ]
        elif args.mode == "step":
            yamls = glob.glob('{}/step_*.yaml'.format(args.src_path))
        else:
            yamls = glob.glob('{}/epoch_*.yaml'.format(args.src_path))
        for y in yamls:
            if "init.yaml" in y or "train.yaml" in y or "snapsnot" in y:
               continue
            print(f"reading {y}")
            with open(y, 'r') as f:
                dic_yaml = yaml.load(f, Loader=yaml.FullLoader)
                # since we have old snapshots with the error
                if args.use_att_acc:
                    if 'att_acc' in dic_yaml:
                        att_acc = dic_yaml['att_acc']
                    elif 'acc_att' in dic_yaml:
                        att_acc = dic_yaml['acc_att']

                loss = dic_yaml['loss_dict']['loss']
                epoch = dic_yaml['epoch']
                step = dic_yaml['step']
                tag = dic_yaml['tag']
                if epoch >= args.min_epoch and epoch <= args.max_epoch \
                        and step >= args.min_step and step <= args.max_step:
                    val_scores += [[epoch, step, loss, tag]]
        sorted_val_scores = sorted(val_scores,
                                   key=lambda x: x[2],
                                   reverse=False)
        print("best val (epoch, step, loss, tag) = " +
              str(sorted_val_scores[:args.num]))
        path_list = [
            args.src_path + '/{}.pt'.format(score[-1])
            for score in sorted_val_scores[:args.num]
        ]
    else:
        if args.from_snapshot:
            path_list = glob.glob('{}/snapshot_*.pt'.format(args.src_path))
        else:
            path_list = glob.glob('{}/[!avg][!final][!snapshot]*.pt'.format(args.src_path))

        path_list = sorted(path_list, key=os.path.getmtime)

    assert len(path_list) == num
    for path in path_list:
        yield torch.load(path, map_location=torch.device('cpu'))


def get_snapshots_wandb(wandb_project, val_best, num, min_epoch, max_epoch, from_snapshot, min_step, max_step, run_tag, run_name):
    """ yields loaded pytorch models to be averaged

    Input:
      wandb_project -- name of wandb project
      val_best -- if True, use validation loss to rank snapshots
      num -- number of models to average
      min_epoch -- minimum epoch of snapshots to select
      max_epoch -- maximum epoch of snapshots to select 
      from_snapshot -- if True, can include mid-epoch snapshots in average instead of only using end-of-epoch models
      use_att_acc -- if True, use attention accuracy on the validation set to rank snapshots
      min_step -- minimum training step of snapshots to select
      max_step -- maximum training step of snapshots to select
      run_tag -- only select snapshots with matching metadata field "run_tag"
      run_name -- only select snapshots from this wandb run
    """
    artifact_filter = defaultdict(dict)
    if min_epoch:
        artifact_filter['metadata.epoch']['$gte'] = min_epoch
    if max_epoch >= 0:
        artifact_filter['metadata.epoch']['$lte'] = max_epoch

    if min_step:
        artifact_filter['metadata.step']['$gte'] = min_step
    if max_step:
        artifact_filter['metadata.step']['$lte'] = max_step

    if val_best:
        artifact_order = 'metadata.loss_dict.loss'
    else:
        artifact_order = '-metadata.loss_dict.acc'
    #print(f"{artifact_order=}")

    if from_snapshot:
        artifact_filter['metadata.name']['$regex'] = '([0-9]+|snapshot)'
    else:
        artifact_filter['metadata.name']['$regex'] = '[0-9]+'

    if run_tag:
        artifact_filter['metadata.run_tag']['$regex'] = run_tag
    if run_name:
        artifact_filter['metadata.wandb_run'] = run_name
 
    t0 = time.time()
    try:
        artifacts = wandb.apis.public.Artifacts(
            client = wandb.Api().client,
            entity = 'pilot',
            project = wandb_project,
            collection_name = "snapshot",
            type = "pytorch_model",
            filters = dict(artifact_filter),
            order = artifact_order
        )[:num]
    except Exception as e:
        logging.error(f"Error retrieving artifacts. Your wandb project {wandb_project} may not exist, or may not contain any artifacts. Error {e}")
        exit(1)
    t1 = time.time()
    print(f"Getting the list of {len(artifacts)} artifacts to merge took {(t1-t0):.3f} seconds")
    for a in artifacts:
        print(f"step {a.metadata['step']:8d}, loss {a.metadata['loss_dict']['loss']:.4f}, accuracy {a.metadata['loss_dict']['acc']:.4f}")

    config = artifacts[0].logged_by().config
    assert len(artifacts) == num
    for a in artifacts:
        a_config = a.logged_by().config 
        assert config == a_config, "not all model artifacts have the same training config"

    run_config = {"training_config": config, 
                  "averaging_config": {"val_best": val_best, "num": num, "min_epoch": min_epoch, "max_epoch": max_epoch, "from_snapshot": from_snapshot}}
    run = wandb.init(config=config, job_type='averaging_pytorch_model', project=wandb_project, entity='pilot')

    def state_dict_generator(artifacts, run):
        for a in artifacts:
            with TemporaryDirectory(prefix="/shared/tmp") as tmpdir:
                t0 = time.time()
                snapshot_path = a.file(root=tmpdir)
                run.use_artifact(a)
                t1 = time.time()
                vv = torch.load(snapshot_path, map_location=torch.device('cpu'))
                t2 = time.time()
                # print(f"\n\ndownloading snapshot took {(t1-t0):.2f} seconds, loading state dict took {(t2-t2):.2f} secs\n\n")
                yield vv

    sd = state_dict_generator(artifacts, run)
    return sd, run 
        

def main():
    args = get_args()
    checkpoints = []
    val_scores = []

    if args.wandb_project:
        state_dict_list, wandb_run = get_snapshots_wandb(args.wandb_project, args.val_best, args.num, args.min_epoch, 
                                                         args.max_epoch, args.from_snapshot, args.min_step, args.max_step, 
                                                         args.run_tag, args.run_name)
    else:
        state_dict_list = get_snapshots_local(args.src_path, args.val_best, args.num, args.min_epoch, args.max_epoch, args.from_snapshot)

    avg = {}
    num = args.num
    dividor = num
    for states in tqdm.tqdm(reversed(list(state_dict_list)), desc="Averaging snapshots"):
        if 'model0' in states:
            states = states['model0']
        t0 = time.time()
        for k in states.keys():
            if k not in avg.keys():
                avg[k] = states[k].clone()
            else:
                avg[k] += states[k]
                #avg[k] = torch.true_divide(4.0*avg[k] + 5.0 * states[k], 9.0)
                #dividor = 1
        t1 = time.time()
        # print(f"traversing that checkpoint took {(t1-t0):.2f} seconds")
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], dividor)
    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)

    if args.wandb_project:
        dst_model_name = Path(args.dst_model).stem
        avg_artifact = wandb.Artifact(dst_model_name, type='pytorch_model')
        avg_artifact.add_file(args.dst_model)
        wandb_run.log_artifact(avg_artifact)
        logging.info(f'averaged model artifact saved as {dst_model_name} in project {args.wandb_project}')


if __name__ == '__main__':
    main()
