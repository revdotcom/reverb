# Copyright 2020 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
import os
import argparse
import glob

import yaml
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path', help='src model path for average')
    parser.add_argument('--list', help='list of snapshots to merge')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    chkpt_paths = []

    if args.list:
        with open(args.list, "r") as reader:
            for line in reader:
                line=line.strip()
                if line[-3:] != ".pt":
                    line = line + ".pt"

                if os.path.isabs(line):
                    chkpt_paths.append(line)
                elif os.path.exists(line):
                    chkpt_paths.append(line)
                elif args.src_path:
                    chkpt_paths.append(args.src_path + "/" + line)

    num=len(chkpt_paths)
    print(f"num ({num}), len(chkpt_paths) = {len(chkpt_paths)}")
    avg = None
    for path in chkpt_paths:
        print('Processing {}'.format(path))
        global_states = torch.load(path, map_location=torch.device('cpu'))
        if 'model0' in global_states:
            states = global_states['model0']
        else:
            states = global_states

        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)


if __name__ == '__main__':
    main()
