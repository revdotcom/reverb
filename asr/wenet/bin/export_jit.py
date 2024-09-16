# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
import logging
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml
import wandb

from wenet.utils.init_model import init_model
from wenet.utils.checkpoint import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=False, help='config file')
    parser.add_argument('--wandb_project', required=False, help='wandb project - can be used instead of a local config file.')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', default=None, help='output file')
    parser.add_argument('--output_quant_file',
                        default=None,
                        help='output quantized model file')
    parser.add_argument('--relax', 
                        action='store_true', 
                        help='Will load the checkpoint with strict=False, useful to updating from previous version of WeNet')

    # parameters for joint decoding - making it easier to experiment for now, in future these should be passed from speech2ctm
    parser.add_argument('--ctc_weight',
                        #default=0.5,
                        help='ctc weight for joint decoding')
    parser.add_argument('--length_bonus',
                        #default=0.5,
                        help='length bonus for joint decoding')
    parser.add_argument('--pre_beam',
                        #default=1.5,
                        help='pre-beam ratio for joint decoding')
    parser.add_argument('--lexicon_file',
                        required=False,       # TODO: make this not required. Export breaks if no lexicon_file supplied.
                        help='lexicon file for joint decoding')
    parser.add_argument('--token_file',
                        required=False,       # TODO: make this not required. Export breaks if no token_file supplied.
                        help='tk.units.txt file for joint decoding')
    args = parser.parse_args()
    return args


def get_local_model(config, checkpoint, args):
    with open(config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # hack-y ways of adding joint decoding parameters
    if args.lexicon_file:
        configs["model_conf"]["lexicon_path"] = args.lexicon_file
    if args.token_file:
        configs["model_conf"]["token_path"] = args.token_file
    # disable the ts_conf portion if it exists, we don't want to export the teacher
    if 'ts_conf' in configs:
        del configs['ts_conf']
    model, configs = init_model(args, configs)
    if args.ctc_weight is not None:
        model.joint_ctc_weight = float(args.ctc_weight)
    if args.length_bonus is not None:
        model.length_bonus = float(args.length_bonus)
    if args.pre_beam is not None:
        model.pre_beam_ratio = float(args.pre_beam)
    model.eval()
    print(model)
    # Export jit torch script model
    load_checkpoint(model, checkpoint, def_strict=not args.relax)
    return model


def get_wandb_model(wandb_project, checkpoint, args):
    print(f"artifact query: {wandb_project}, {checkpoint}")
    try:
        artifacts = wandb.apis.public.ArtifactVersions(
            client = wandb.Api().client,
            entity = "pilot",
            project = wandb_project,
            collection_name = checkpoint,
            type = "pytorch_model"
        )
        if len(artifacts) > 1: # using the @latest artifact
            for a in artifacts:
                if 'latest' in a.aliases:
                    artifact = a
                    break
        elif len(artifacts) == 0:
            raise Exception
        else:
            artifact = artifacts[0]
    except Exception:
        logging.error(f"Error retrieving artifacts. Your wandb project {wandb_project} may not exist, or may not contain any artifact named {checkpoint}.")
        exit()
    config = artifact.logged_by().config
    if 'training_config' in config:
        config = config['training_config']
    if 'ts_conf' in config:
        del config['ts_conf']
    run_config = {"training_config": config, "export_config": vars(args)}
    run = wandb.init(config=run_config, project=wandb_project, job_type='export_pt_to_torchscript', entity='pilot')
    print(f"using artifact {artifact.name} {artifact.version}")
    run.use_artifact(artifact)
    model = init_model(args, config)
    model.joint_ctc_weight = float(args.ctc_weight)
    model.length_bonus = float(args.length_bonus)
    model.pre_beam_ratio = float(args.pre_beam)
    model.eval()
    with TemporaryDirectory(prefix="/shared/tmp") as tempdir:
        snapshot_path = artifact.file(root=tempdir)
        load_checkpoint(model, snapshot_path, def_strict=not args.relax)
        return model, run



def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.wandb_project:
        model, wandb_run = get_wandb_model(args.wandb_project, args.checkpoint, args)
    else:
        model = get_local_model(args.config, args.checkpoint, args)

    if args.output_file:
        script_model = torch.jit.script(model)
        script_model.save(args.output_file)
        print('Export model successfully, see {}'.format(args.output_file))

    if args.wandb_project and args.output_file:
        output_name = Path(args.output_file).stem
        artifact = wandb.Artifact(name=output_name, type='torchscript_model', metadata={'quantized': False})
        artifact.add_file(args.output_file)
        wandb_run.log_artifact(artifact)

    # Export quantized jit torch script model
    if args.output_quant_file:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
        script_quant_model = torch.jit.script(quantized_model)
        script_quant_model.save(args.output_quant_file)
        
    if args.output_quant_file and args.wandb_project:
        output_name = Path(args.output_quant_file).stem
        artifact = wandb.Artifact(name=output_name, type='torchscript_model', metadata={'quantized': True})
        artifact.add_file(args.output_file)
        wandb_run.log_artifact(artifact)
        print('Export quantized model successfully, '
              'see {}'.format(args.output_quant_file))


if __name__ == '__main__':
    main()
