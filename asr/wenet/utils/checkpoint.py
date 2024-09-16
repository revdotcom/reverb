# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
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

import logging
import os
import io
import re
import wandb

import yaml
import torch
from collections import OrderedDict

import datetime
import shutil


def load_checkpoint(model: torch.nn.Module, path: str, encoder_only: bool=False, force_cpu: bool=True, optimizer: torch.nn.Module=None, def_strict=True) -> dict:
    if not force_cpu and torch.cuda.is_available():
        logging.debug('Checkpoint: loading from checkpoint %s for GPU' % path)
        global_checkpoint = torch.load(path)
    else:
        logging.debug('Checkpoint: loading from checkpoint %s for CPU' % path)
        global_checkpoint = torch.load(path, map_location='cpu')


    if 'model0' in global_checkpoint:
        checkpoint = global_checkpoint['model0']
    else:
        checkpoint = global_checkpoint

    espnet_model = False
    try:
        #loading ESPnet model
        #Missing key(s) in state_dict: "encoder.global_cmvn.mean", "encoder.global_cmvn.istd".
        #Unexpected key(s) in state_dict: "normalize.mean", "normalize.std".
        checkpoint["encoder.global_cmvn.mean"] = checkpoint["normalize.mean"]
        checkpoint["encoder.global_cmvn.istd"] = checkpoint["normalize.std"]
        del checkpoint["normalize.mean"]
        del checkpoint["normalize.std"]
        espnet_model = True
        logging.info("ESPNET model detected")
    except:
        pass

    if encoder_only:
       for k in list(checkpoint.keys()):
           if 'encoder.' not in k:
               print(f"encoder_only is True, deleting {k}")
               del checkpoint[k]

    strict = def_strict and (not encoder_only and not espnet_model)
    model.load_state_dict(checkpoint, strict=strict)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}

    if optimizer is not None:
        if 'optimizer0' in global_checkpoint:
            optimizer.load_state_dict(global_checkpoint['optimizer0'])
            logging.info("optimizer restored from checkpoint")
        else:
            logging.warning('optimizer0 not found in checkpoint')
    else:
        logging.info("optimizer state won't be restored")

    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs

def upstream_save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)

def write_checkpoint_yaml(model_file_path : str, infos : dict = None, optimizer = None) :
    info_path = re.sub('.pt$', '.yaml', model_file_path)
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    infos['includes_optimizer'] = optimizer is not None

    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)

def save_checkpoint(model: torch.nn.Module, model_dir: str, infos=None, snapshot_conf: dict = None, snapshot_name: str = "snapshot", optimizer: torch.nn.Module=None ):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    if infos is None:
       infos = {}

    if 'ts_conf' in infos:
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        if hasattr(model, 'student'):
            model = model.student

    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # if 'student' in model_state_dict:
    #     model_state_dict = model_state_dict['student']

    overall_dict = dict()
    overall_dict["model0"] = model_state_dict

    if optimizer is not None:
        overall_dict["optimizer0"] = optimizer.state_dict()
        logging.info("optimizer saved to checkpoint")
    else:
        logging.info("optimizer *not* saved to checkpoint")

    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    infos['includes_optimizer'] = optimizer is not None
    infos['run_name'] = wandb.run.name
    if snapshot_conf.get('run_tag'):
        infos['run_tag'] = snapshot_conf.get('run_tag')

    # Determining path for local save
    # Option 1: Saving specially named models:
    #        1a: Save all "epoch_.*" models
    #        1b: If `snapshot_conf` has `use_named_snapshots` set to True
    #            use the `snapshot_name`
    if "epoch_" in snapshot_name or snapshot_conf is not None and snapshot_conf.get('use_named_snapshots', True):
        path = f"{model_dir}/{snapshot_name}.pt"
    # Option 2: If the optimizer is set, let's make it obvious in the name
    elif optimizer is not None:
        path = f"{model_dir}/snapshot_and_optimizer.pt"
    # Option 3: Default the local model to "snapsho.pt"
    else:
        path = f"{model_dir}/snapshot.pt"

    logging.info('Checkpoint: save to checkpoint %s' % path)
    torch.save(overall_dict, path)

    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    infos['includes_optimizer'] = optimizer is not None

    write_checkpoint_yaml(path, infos, optimizer)

    # JPR : save to wandb
    # if save to wandb
    if wandb.run is not None:
        if snapshot_conf is not None and snapshot_conf.get('save_to_wandb', True):
            infos['name'] = snapshot_name
            if snapshot_conf.get('run_tag'):
                infos['run_tag'] = snapshot_conf.get('run_tag')

            wandb_runname = wandb.run.name
            infos['wandb_run'] = wandb_runname
            snapshot_artifact = wandb.Artifact('snapshot', type="pytorch_model", metadata=infos)
            snapshot_artifact.add_file(path, name='snapshot.pt')
            wandb.log_artifact(snapshot_artifact)

def delete_forced_full_snapshot_flag(model_dir, rank):
    flag_file = os.path.join(model_dir, 'force_full_snapshot')
    if os.path.exists(flag_file) and rank == 0:
        os.remove(flag_file)

def filter_modules(model_state_dict, modules):
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods:
        logging.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                     enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location='cpu')
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        logging.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs

def check_forced_full_snapshot_flag(model_dir, batch_idx = -1):
    """Will check for the existence of a file called force_full_snapshot in the model_dir.
        If it exists, it will return True if the file is empty or if the batch_idx is below 0
        or if batch_idx is greater than the value in the file.
        If conditions are met, the file will be deleted.

        If the file content is not a valid integer, this will return true and the file will be deleted."""
    force_full_snapshot = False
    flag_file = os.path.join(model_dir, 'force_full_snapshot')
    if os.path.exists(flag_file):
        force_full_snapshot = True
        if batch_idx > 0:
            try:
                val = open(flag_file).read().strip()
                val = int(val)
                if batch_idx < val:
                    force_full_snapshot = False
            except:
                val = -1

        if force_full_snapshot:
            logging.info(f"found {flag_file}, will force a full snapshot")

    return force_full_snapshot

def download_checkpoint_from_wandb(
    model: torch.nn.Module,
    path: str,
) -> str:
    if wandb.run is None:
        raise RuntimeError("Can't find checkpoint from WandB since it hasn't been initialized. Failing!")

    latest_optimizer_artifacts = wandb.apis.public.Artifacts(
        client=wandb.Api().client,
        entity=wandb.run.entity,
        project=wandb.run.project,
        collection_name="snapshot",
        type="pytorch_model",
        filters={"metadata.includes_optimizer": True},
    )
    logging.info(f"Found {len(latest_optimizer_artifacts)} snapshot(s) with optimizer")
    if len(latest_optimizer_artifacts) == 0:
        raise RuntimeError("Can't find checkpoint from WandB since no snapshot with optimizer has been saved. Failing!")

    # These are ordered from most to least recently added
    wandb_snapshot_dir = latest_optimizer_artifacts[0].download()
    shutil.move(os.path.join(wandb_snapshot_dir, "snapshot.pt"), path)
    logging.info(f"Saving WandB snapshot to {path}")

    return path

