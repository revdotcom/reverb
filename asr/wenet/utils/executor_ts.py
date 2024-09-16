# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
from contextlib import nullcontext
# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
import torch.distributed as dist
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint
import wenet.dataset
from torch.nn.utils import clip_grad_norm_
from torch.distributed.algorithms.join import Join, Joinable
import os
import wandb


class TeacherStudentExecutor:
    def __init__(self):
        self.step = 0

    # We've added rank, world_size so that we know when to run CV (will just run on one GPU)
    # cv_data loader and configs are there so that we can run one CV pass as part of the training loop
    # model_dir is there so that we know where to save the snapshot
    # epoch is there to save that info in the yaml file that goes with the snapshot
    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler, rank, world_size, cv_data_loader, model_dir, configs, epoch):
        ''' Train one epoch
        '''

        is_distributed = args.get('is_distributed', True)

        if is_distributed:
            model.module.student.train()
        else:
            model.student.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        freeze_encoder = args.get('freeze_encoder', False)
        if freeze_encoder:
            print("freezing the encoder")
            for name, param in model.named_parameters():
                if param.requires_grad and 'encoder.' in name:
                    param.requires_grad = False

        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            #model_context = model.join
            parallel_objects = [model]
            if isinstance(optimizer, Joinable):
                parallel_objects.append(optimizer)
            model_context = Join(parallel_objects, throw_on_early_termination=configs.get('throw_on_early_termination', False))
        else:
            model_context = nullcontext()
        num_seen_utts = 0
        # num_total_batch = len(data_loader)
        # snap_every = num_total_batch // 25 
        snap_every=args.get('snap_every', 1235)
        logging.info(f"snap-every = {snap_every}")
        try:
            with model_context:
                for batch_idx, batch in enumerate(data_loader):
                    if True and batch_idx >= 50000:
                        # ugly hack to see if we can move over one epoch
                        continue
                    key, feats, target, feats_lengths, target_lengths = batch
                    feats = feats.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    # logging.debug(f"rank {rank}, feats.shape {feats.shape}, target shape {target.shape}")

                    feats_lengths = feats_lengths.to(device, non_blocking=True)
                    target_lengths = target_lengths.to(device, non_blocking=True)
                    num_utts = target_lengths.size(0)
                    if num_utts == 0:
                        continue
                    context = None
                    # Disable gradient synchronizations across DDP processes.
                    # Within this context, gradients will be accumulated on module
                    # variables, which will later be synchronized.
                    if is_distributed and batch_idx % accum_grad != 0:
                        context = model.no_sync
                    # Used for single gpu training and DDP gradient synchronization
                    # processes.
                    else:
                        context = nullcontext
                    with context():
                        # autocast context
                        # The more details about amp can be found in
                        # https://pytorch.org/docs/stable/notes/amp_examples.html
                        with torch.cuda.amp.autocast(scaler is not None):
                            # loss, loss_att, loss_ctc, acc_att = model(
                            #     feats, feats_lengths, target, target_lengths)

                            losses_and_other = model(feats, feats_lengths, target, target_lengths)
                            # "loss":ts_loss, 'mse_loss':mse_loss, "student_loss":loss, "loss_att":loss_att, "loss_ctc": loss_ctc, "acc_att": acc_att}
                            loss = losses_and_other["loss"]
                            loss_kl_enc = losses_and_other["kl_enc_loss"]
                            loss_kl_dec = losses_and_other["kl_dec_loss"]
                            loss_att = losses_and_other["loss_att"]
                            loss_ctc = losses_and_other["loss_ctc"]
                            acc_att = losses_and_other["acc_att"]
                            ts_weight = losses_and_other["ts_weight"]
    
                            loss = loss / accum_grad
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    num_seen_utts += num_utts
                    if batch_idx % accum_grad == 0:
                        if rank == 0 and writer is not None:
                            writer.add_scalar('train_loss', loss, self.step)

                            if loss_ctc :
                                writer.add_scalar('train_ctc_loss', loss_ctc, self.step)
                            if loss_att:
                                writer.add_scalar('train_att_loss', loss_att, self.step)
                            if acc_att >= 0:
                                writer.add_scalars('acc', {'att': acc_att}, self.step)
                        # Use mixed precision training
                        if use_amp:
                            scaler.unscale_(optimizer)
                            grad_norm = clip_grad_norm_(model.parameters(), clip)
                            # Must invoke scaler.update() if unscale_() is used in
                            # the iteration to avoid the following error:
                            #   RuntimeError: unscale_() has already been called
                            #   on this optimizer since the last update().
                            # We don't check grad here since that if the gradient
                            # has inf/nan values, scaler.step will skip
                            # optimizer.step().
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_norm = clip_grad_norm_(model.parameters(), clip)
                            if torch.isfinite(grad_norm):
                                optimizer.step()
                            else:
                                logging.warn(f"optimizer step skipped because gradiants were infinite, batch_idx {batch_idx}, rank {rank}")

                        if torch.isfinite(grad_norm):
                            scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        self.step += 1
                    if batch_idx % log_interval == 0:
                        lr = optimizer.param_groups[0]['lr']
                        log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                            epoch, batch_idx,
                            loss.item() * accum_grad)
                        # if loss.item() > 100:
                        #     print(f"loss > 100 for key: {key}, batch text {target}, text len {target_lengths}, input len {feats_lengths}")
                        wandb_log = {'loss': loss.item() * accum_grad}

                        if loss_kl_enc is not None:
                            log_str += 'loss_kl_enc {:.6f} '.format(loss_kl_enc.item())
                            wandb_log['kl/enc_loss'] = loss_kl_enc.item()

                        if ts_weight is not None:
                            log_str += 'ts_weight {:.6f} '.format(ts_weight)
                            wandb_log['kl/ts_weight'] = ts_weight
    
                        if loss_kl_dec is not None:
                            log_str += 'loss_kl_dec {:.6f} '.format(loss_kl_dec.item())
                            wandb_log['kl/dec_loss'] = loss_kl_dec.item()
    
                        if loss_att is not None:
                            log_str += 'loss_att {:.6f} '.format(loss_att.item())
                            log_str += 'acc_att {:.6f} '.format(acc_att)
                            wandb_log['loss_att'] = loss_att.item()
                            wandb_log['acc_att'] = acc_att

                        if loss_ctc is not None:
                            log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                            wandb_log['loss_ctc'] = loss_ctc.item()

                        log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                        if rank == 0:
                            wandb.log(wandb_log)
                        logging.debug(log_str)
                    # create a snapshot that would allow us to restart training more easily
                    # if we ever use spot instances
                    if batch_idx > 0 and batch_idx % snap_every == 0:
                        total_loss, total_att_acc, cv_num_seen_utts = self.cv(model, cv_data_loader, device,
                                                                configs)
                        logging.debug(f"rank {rank}, filter stats {wenet.dataset.processor.mystats}")
                        logging.debug(f"rank {rank}, filter stats rev_processor_ex {wenet.dataset.rev_processor_ex.mystats}")

                        if world_size > 1:
                            # all_reduce expected a sequence parameter, so we use [num_seen_utts].
                            cv_num_seen_utts = torch.Tensor([cv_num_seen_utts]).to(device, non_blocking=True)
                            # the default operator in all_reduce function is sum.
                            dist.all_reduce(cv_num_seen_utts)
                            total_loss = torch.Tensor([total_loss]).to(device, non_blocking=True)
                            dist.all_reduce(total_loss)
                            cv_loss = total_loss[0] / cv_num_seen_utts[0]
                            cv_loss = cv_loss.item()

                            total_att_acc = torch.Tensor([total_att_acc]).to(device, non_blocking=True)
                            dist.all_reduce(total_att_acc)
                            acc_att = total_att_acc[0] / cv_num_seen_utts[0]
                            acc_att = acc_att.item()

                        else:
                            cv_loss = total_loss / cv_num_seen_utts
                            acc_att = total_att_acc / cv_num_seen_utts

                        lr = optimizer.param_groups[0]['lr']
                        logging.info(f"{batch_idx} -> CV LOSS = {cv_loss})")
                        if rank == 0:
                            logging.info(f"taking a snapshot at {epoch}/{batch_idx})")
                            writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr, 'epoch' : epoch}, self.step)
                            wandb_log = {'cv_loss': cv_loss, 'lr': lr, 'epoch' : epoch, 'cv_acc_att': acc_att }
                            wandb.log(wandb_log)
                            save_model_path = os.path.join(model_dir, 'snapshot.pt')
                            save_checkpoint(
                            
                                model.module.student if is_distributed else model.student, save_model_path, {
                                    'epoch': epoch,
                                    'lr': lr,
                                    'cv_loss': cv_loss,
                                    'acc_att': acc_att,
                                    'step': self.step
                                })
                            # let's try a bit of cleanup
                            del acc_att,  total_att_acc, cv_num_seen_utts, total_loss, cv_loss

                        # let's set the model back in training mode
                        if is_distributed:
                            model.module.student.train()
                        else:
                            model.student.train()

                        if freeze_encoder:
                            print("freezing the encoder")
                            for name, param in model.named_parameters():
                                if param.requires_grad and 'encoder.' in name:
                                    param.requires_grad = False
        except RuntimeError as err:
            optimizer.zero_grad(set_to_none=True)
            logging.warning(f"Early Termination: Saw {num_seen_utts} utts by rank {rank}")
        finally:
            if is_distributed:
                dist.barrier()

    @torch.no_grad()
    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        total_att_acc = 0.0
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.no_sync
        else:
            model_context = nullcontext
        with model_context() as MC, torch.no_grad() as NG:
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                feats = feats.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                feats_lengths = feats_lengths.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)

                losses_and_other = model(feats, feats_lengths, target, target_lengths)
                loss = losses_and_other["loss"]
                loss_att = losses_and_other["loss_att"]
                loss_ctc = losses_and_other["loss_ctc"]
                acc_att = losses_and_other["acc_att"]

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                    total_att_acc += acc_att * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                        log_str += 'acc_att {:.6f} '.format(acc_att)
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
        return total_loss, total_att_acc, num_seen_utts
