# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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

import tracemalloc

import copy
import datetime
import logging
import sys
from contextlib import nullcontext
import time

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
import math

import torch.distributed as dist
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint, check_forced_full_snapshot_flag, delete_forced_full_snapshot_flag
import wenet.dataset
from wenet.utils.train_utils import (wenet_join, batch_forward, batch_backward,
                                     update_parameter_and_lr, log_per_step,
                                     save_model)
import os
import wandb
import re
import gc

#from pympler import muppy
#from memory_profiler import profile
#mem_logs = open('mem_profile.log', 'a')

class Executor:

    def __init__(self):
        self.step = 0
        self.num_seen_frames : int = 0

    #@profile(stream=mem_logs)
    def train(self, model, optimizer, scheduler, train_data_loader,
              cv_data_loader, writer, configs, scaler, group_join, cmdline_args, epoch):
        ''' Train one epoch
        '''
        model.train()

        # JPR : Note, these three blocs (free_encoder, freeze_non_lsl and learning_re_rules 
        #       are overlapping each-other.  We should probably revisit this later.
        freeze_encoder = cmdline_args.freeze_encoder
        freeze_non_lsl = cmdline_args.freeze_non_lsl
        learning_re_rules = configs.get('restrict_learning', {})
        deep_biasing = configs['dataset_conf'].get('deep_bias_conf', {}).get('deep_biasing', False)


        # JPR : Note, these three blocs (free_encoder, freeze_non_lsl and learning_re_rules 
        #       are overlapping each-other.  We should probably revisit this later.
        if freeze_encoder:
            self.FreezeEncoder(model)

        if freeze_non_lsl:
            logging.info("freezing everything except lsls")            
            for name, param in model.named_parameters():
                if param.requires_grad and not 'language' in name:
                    param.requires_grad = False

        if deep_biasing:
            logging.info("freezing everything except context adaptor")
            for name, param in model.named_parameters():
                if param.requires_grad and not 'adaptor' in name:
                    param.requires_grad = False
            

        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        is_distributed = True if world_size > 1 else False

        if learning_re_rules:
            self.SetupLearningFlags(model, learning_re_rules, verbose=(rank == 0 and epoch == 0))

        info_dict = copy.deepcopy(configs)
        info_dict['epoch'] = epoch

        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # model_context = model.join(throw_on_early_termination=configs.get('throw_on_early_termination', True))
            model_context = model.join
        else:
            model_context = nullcontext

        # JPR: bringing some stuff from rev-wenet
        device = int(os.environ.get('LOCAL_RANK', 0))
        num_seen_utts = 0
        snapshot_conf = configs['snapshot_saving_conf']
        # snap_every = snapshot_conf.get('snap_every', 3000)
        log_interval = configs.get('log_interval', 100)
        save_optimizer_every = snapshot_conf.get('save_optimizer_every', None)
        named_snapshots = snapshot_conf.get('use_named_snapshots', False)

        accum_grad = info_dict['accum_grad']
        # save_interval = info_dict.get('save_interval', sys.maxsize)
        save_interval  = snapshot_conf.get('save_interval', int(3000/accum_grad))
        save_optimizer_every = snapshot_conf.get('save_optimizer_every', 10)
        prev_snapshot = None
        model_dir = info_dict["model_dir"]

        try:
            with model_context(throw_on_early_termination=True):
                local_seen_frames = torch.tensor(0, dtype=torch.long, device=device)
                # last_objects = gc.get_objects(generation=None)
                # last_objects = muppy.get_objects()
                for batch_idx, batch_dict in enumerate(train_data_loader):
                    info_dict["tag"] = "TRAIN"
                    info_dict["step"] = self.step
                    info_dict["batch_idx"] = batch_idx

                    # if rank == 0:
                    #     for k in batch_dict.keys():
                    #         print(f"batch_dict[{k}] = {type(batch_dict[k])}")
                        

                    if wenet_join(group_join, info_dict):
                        break

                    #JPR: Let's bring this around, for debugging purposes
                    if False and batch_idx >= 50000:
                        # JPR: ugly hack to see if we can move over one epoch
                        # it is useful to keep this here in case one wants to tests quickly 
                        # despite some problem with the barriers
                        break


                    if batch_dict["target_lengths"].size(0) == 0:
                        logging.warn(f"on rank {rank} we got target_length.size(0) == 0")
                        continue

                    local_seen_frames += batch_dict['feats_lengths'].sum().detach()

                    #if batch_idx <= 600 and batch_idx <= 259900:
                    #    if batch_idx % log_interval == 0:
                    #        #dist.barrier()
                    #        logging.info(f"Train - skip {batch_idx} rank {rank}")
                    #    if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    #        dist.barrier()
                    #    continue
                    #else:
                    #    logging.info(f"setting log_interval = 1 rank {rank}")
                    #    #dist.barrier()
                    #    log_interval = 1

                    context = None
                    # Disable gradient synchronizations across DDP processes.
                    # Within this context, gradients will be accumulated on module
                    # variables, which will later be synchronized.
                    if info_dict.get("train_engine", "torch_ddp") == "torch_ddp" and \
                            (batch_idx + 1) % info_dict["accum_grad"] != 0:
                        context = model.no_sync
                    # Used for single gpu training and DDP gradient synchronization
                    # processes.
                    else:
                        context = nullcontext

                    with context():
                        info_dict = batch_forward(model, batch_dict, scaler,
                                                  info_dict)
                        info_dict = batch_backward(model, scaler, info_dict)

                        info_dict = update_parameter_and_lr(model, optimizer,
                                                            scheduler, scaler,
                                                            info_dict)

                        # TODO : should perhaps move to log_per_scrip
                        time_updated_now = False
                        if batch_idx % log_interval == 0:
                            total_frames_so_far = self.update_seen_frames_if_needed(is_distributed, rank, local_seen_frames)
                            time_updated_now = True
                            info_dict['num_seen_frames'] = total_frames_so_far
                            local_seen_frames.zero_()
                            log_per_step(writer, info_dict)
                            # torch.cuda.empty_cache()
                            # local_objects = muppy.get_objects()

                            # let's check if we need to force a full snapshot
                            force_full_snapshot = check_forced_full_snapshot_flag(model_dir, batch_idx)
 
                            # if batch_idx > 100 and rank == 0:
                            #     # Step 3: Compare the two sets to find new objects
                            #     # new_objects = set(local_objects) - set(last_objects)
                            #     # new_objects = muppy.get_diff(last_objects, local_objects)
                            #     # Print the new objects
                            #     for obj in new_objects['+']:
                            #         print(obj) 
                            # last_objects = local_objects

                        time_to_save = (self.step + 1) % save_interval == 0 and self.step != 0
                        time_to_save = time_to_save or force_full_snapshot
                        # time_to_save = time_to_save or (self.step + 1) % snap_every == 0
                        if time_to_save and (batch_idx + 1) % info_dict["accum_grad"] == 0:
                            if not time_updated_now:
                                total_frames_so_far = self.update_seen_frames_if_needed(is_distributed, rank, local_seen_frames)
                                info_dict['num_seen_frames'] = total_frames_so_far
                                local_seen_frames.zero_()
                            loss_dict = self.cv(model, cv_data_loader, configs)
                            model.train()
                            info_dict.update({
                                "tag":
                                f"step_{self.step:09d}",
                                "loss_dict":
                                loss_dict,
                                "save_time":
                                datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                                "lr":
                                optimizer.param_groups[0]['lr']
                            })
                            # info_dict['frames_seen_so_far'] = 0
                            if force_full_snapshot or (self.step + 1) % (save_interval * save_optimizer_every) == 0:
                                save_model(model, info_dict, optimizer=optimizer)
                                if force_full_snapshot:
                                    delete_forced_full_snapshot_flag(model_dir, rank)
                            else:
                                save_model(model, info_dict)
                        log_per_step(writer, info_dict)
                        self.step += 1 if (batch_idx + 1) % info_dict["accum_grad"] == 0 else 0

        #except Exception as ex:
        except RuntimeError as err:
            optimizer.zero_grad(set_to_none=True)
            logging.info(f"executor.py: early stopping {err} {rank}")
        finally:
            logging.info(f"executor.py: done looping train_data_loader rank {rank}")
            if is_distributed:
                dist.barrier()
                total_frames_so_far = self.update_seen_frames_if_needed(is_distributed, rank, local_seen_frames)

    def cv(self, model, cv_data_loader, configs):
        ''' Cross validation on
        '''
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        logging.debug(f"Starting cross-validation/DEV work on local_rank {local_rank}")
        model.eval()
        info_dict = copy.deepcopy(configs)
        num_seen_utts, loss_dict, total_acc = 1, {}, []  # avoid division by 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(cv_data_loader):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx

                num_utts = batch_dict["target_lengths"].size(0)
                if num_utts == 0:
                    continue

                info_dict = batch_forward(model, batch_dict, None, info_dict)
                _dict = info_dict["loss_dict"]

                num_seen_utts += num_utts
                total_acc.append(_dict['th_accuracy'].item(
                ) if 'th_accuracy' in _dict and _dict['th_accuracy'] is not None else 0.0)
                for loss_name, loss_value in _dict.items():
                    if loss_value is not None and "loss" in loss_name :
                        if type(loss_value) is float and math.isfinite(loss_value):
                            loss_value = loss_value
                        elif torch.isfinite(loss_value):
                            loss_value = loss_value.item()
                        loss_dict[loss_name] = loss_dict.get(loss_name, 0) + \
                            loss_value * num_utts

                log_per_step(writer=None, info_dict=info_dict)
        for loss_name, loss_value in loss_dict.items():
            loss_dict[loss_name] = loss_dict[loss_name] / num_seen_utts
        loss_dict["acc"] = sum(total_acc) / len(total_acc)
        return loss_dict

    def FreezeEncoder(self, model):
        print("freezing the encoder")
        for name, param in model.named_parameters():
            if param.requires_grad and 'encoder.' in name:
                param.requires_grad = False


    # JPR : from rev-wenet
    def SetupLearningFlags(self, model, rulesList, verbose=False):
        """This method will interpret the rules from the restrict_learning yaml section.
            Rules to include or exclude a layer from the set of learnable parameters are 
            processed in the order they appear in the yaml file.
        """

        for name, param in model.named_parameters():
            if param.requires_grad :
                requires = True
                for rule_dict in rulesList:
                    key, value = list(rule_dict.items())[0]

                    if key == 'include' and requires:
                        # this parameter is already flag as "yes", no need to evaluate
                        continue
                    if key == 'exclude' and not requires:
                        continue

                    re_expr = value
                    if type(re_expr) is str:
                        re_expr = re.compile(value)
                        rule_dict[key] = re_expr

                    m = re_expr.match(name)
                    if m:
                        requires = not requires
                if verbose:
                    logging.info(f"in the end, {name} will have requires_grad = {requires}")
                param.requires_grad = requires

    def update_seen_frames_if_needed(self, is_distributed, rank, local_seen_frames):
        """This method will update the total number of frames seen so far, across GPUs if needed.
           It is the responsability of the caller to zero-out local_seen_frames"""
        if is_distributed:
            dist.reduce(local_seen_frames, op=dist.ReduceOp.SUM, dst=0)

        if rank == 0:
            total_frames_so_far : int = self.num_seen_frames + local_seen_frames.item()
            self.num_seen_frames = total_frames_so_far
        return self.num_seen_frames

