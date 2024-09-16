# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.cmvn import load_cmvn
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--length_penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""")
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    parser.add_argument(
        '--overwrite_cmvn',
        action='store_true',
        help='overwrite CMVN params in model with those in config file')

    parser.add_argument('--cat_embs', type=str, default="")
    parser.add_argument('--force_lid_hot', action='append', default=[])

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        default='',
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=0.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')

    parser.add_argument("--use_lora",
                         action="store_true",
                         default=False,
                         help='''Whether to use lora layers''')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    if not 'filter_conf' in test_conf:
        test_conf['filter_conf'] = {}
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['apply_rir'] = False
    test_conf['apply_telephony'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False

    if 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    elif not 'fbank_conf' in test_conf:
        test_conf['fbank_conf'] = {
            "num_mel_bins": 80,
            "frame_shift": 10,
            "frame_length": 25
        }
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0

    if not 'batch_conf' in test_conf:
        test_conf['batch_conf'] = {}
    test_conf['batch_conf']['batch_size'] = args.batch_size
    test_conf['batch_conf']['batch_type'] = "static"
    if not 'cat_emb_conf' in test_conf:
        test_conf['cat_emb_conf'] = {}
    test_conf['cat_emb_conf']['force_hot'] = args.force_lid_hot
    test_conf['cat_emb_conf']['multi_hot'] = False

    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    configs['output_dim'] = len(tokenizer.symbol_table)

    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs)

    # from merge section (JPR)
    if args.overwrite_cmvn and (configs['cmvn_file'] is not None):
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
        model.encoder.global_cmvn = global_cmvn

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    context_graph = None
    if 'decoding-graph' in args.context_bias_mode:
        context_graph = ContextGraph(args.context_list_path,
                                     tokenizer.symbol_table,
                                     configs['tokenizer_conf']['bpe_path'],
                                     args.context_graph_score)

    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    logging.info("blank_id is {}".format(blank_id))

    # TODO(Dinghao Zhou): Support RNN-T related decoding
    # TODO(Lv Xiang): Support k2 related decoding
    # TODO(Kaixun Huang): Support context graph
    files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')
        files[mode] = open(file_name, 'w')
    max_format_len = max([len(mode) for mode in args.modes])
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            keys = batch["keys"]
            feats = batch["feats"].to(device)
            target = batch["target"].to(device)

            # script argument overrides categories in shard
            if len(args.cat_embs) > 0:
                cat_embs = torch.tensor(
                    [float(c) for c in args.cat_embs.split(',')]).to(device)
            elif "cat_emb" in batch:
                cat_embs = batch["cat_emb"]
            else:
                cat_embs = torch.tensor([1] + [0] *
                                        (len(model.cat_labels) - 1)).to(device)

            feats_lengths = batch["feats_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
 # luminary example CV
            # nTerms x maxlen
            cv = torch.zeros(9, 6, dtype=int)
            cv_lengths = torch.tensor([2, 6, 3, 3, 2, 5, 2, 3, 4], dtype=int)
            
            cv[0, 0] = 296
            cv[0, 1] = 7110 #mayzie
            cv[1, 0] = 47
            cv[1, 1] = 1028
            cv[1, 2] = 35
            cv[1, 3] = 81
            cv[1, 4] = 2754
            cv[1, 5] = 710 #weiss-berman
            cv[2, 0] = 63
            cv[2, 1] = 35
            cv[2, 2] = 2947 #doshi
            cv[3, 0] = 34
            cv[3, 1] = 67
            cv[3, 2] = 4299 #imrie
            cv[4, 0] = 1709
            cv[4, 1] = 406 #maron
            cv[5, 0] = 6379
            cv[5, 1] = 1858
            cv[5, 2] = 120
            cv[5, 3] = 1660
            cv[5, 4] = 1169 #glenfiddich
            cv[6, 0] = 2553
            cv[6, 1] = 6165 #levar
            cv[7, 0] = 145
            cv[7, 1] = 1048
            cv[7, 2] = 1645 #cianci
            cv[8, 0] = 1998
            cv[8, 1] = 1703
            cv[8, 2] = 3023
            cv[8, 3] = 314 #bijou house

            infos = {"tasks": batch["tasks"], "langs": batch["langs"]}
            results = model.decode(
                args.modes,
                feats,
                feats_lengths,
                args.beam_size,
                decoding_chunk_size=args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                ctc_weight=args.ctc_weight,
                simulate_streaming=args.simulate_streaming,
                reverse_weight=args.reverse_weight,
                context_graph=context_graph,
                blank_id=blank_id,
                blank_penalty=args.blank_penalty,
                length_penalty=args.length_penalty,
                infos=infos,
                cat_embs=cat_embs,
                cv=cv, 
                cv_lengths=cv_lengths)
            for i, key in enumerate(keys):
                for mode, hyps in results.items():
                    tokens = hyps[i].tokens
                    line = '{} {}'.format(key, tokenizer.detokenize(tokens)[0])
                    logging.info('{} {}'.format(mode.ljust(max_format_len),
                                                line))
                    files[mode].write(line + '\n')
    for mode, f in files.items():
        f.close()


if __name__ == '__main__':
    main()
