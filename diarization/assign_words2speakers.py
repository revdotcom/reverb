#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024
# Author: Jan Profant <jan.profant@rev.com>
# All Rights Reserved

import argparse
import csv
from collections import defaultdict

from intervaltree import IntervalTree, Interval

from pyannote.database.util import load_rttm


def read_ctm(ctm_path):
    with open(ctm_path, 'r') as f:
        csv_file = csv.reader(f, delimiter=' ')
        for row in csv_file:
            yield row


def speaker_for_segment(start: float,
                        dur: float,
                        tree: IntervalTree) -> str:
    """Given a start and duration in seconds, and an interval tree representing
    speaker segments, return what speaker is speaking.

    If there are overlapping speakers, return the speaker who spoke most of the
    time. If there are no speakers, return the nearest one.

    The interval tree could represent reference or hypothesis.
    The data inside the interval tree should be the speaker label.
    """
    intervals = tree[start:start + dur]

    # Easy case, only one possible interval
    if len(intervals) == 1:
        return intervals.pop().data

    # First special case, no match
    # so we need to find the nearest interval
    elif len(intervals) == 0:
        seg = Interval(start, start + dur)
        distances = {interval: seg.distance_to(interval)
                     for interval in tree}
        if not distances:
            return ""
        return min(distances, key=distances.get).data

    # Second special case, overlapping speakers
    # so we return whichever speaker has majority
    else:
        seg = Interval(start, start + dur)
        overlap_sizes = defaultdict(int)
        for interval in intervals:
            i0 = max(seg[0], interval[0])
            i1 = min(seg[1], interval[1])
            overlap_sizes[interval.data] += i1 - i0
        return max(overlap_sizes, key=overlap_sizes.get)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Assign words to speakers based on a diarization rttm file and ctm transcription')
    parser.add_argument('diarization_rttm', help='diarization rttm file')
    parser.add_argument('ctm_transcription', help='ctm transcription file')
    """ Read more about stm format here (we can't store speaker identities in ctm) """
    """ https://www.nist.gov/system/files/documents/2021/08/31/OpenASR21_EvalPlan_v1_3_1.pdf """
    parser.add_argument('output_stm_transcription', help='output file in .stm format as described above')

    args = parser.parse_args()

    ctm = read_ctm(args.ctm_transcription)
    rttm = load_rttm(args.diarization_rttm)
    dict_key = list(rttm.keys())
    assert len(dict_key) == 1, dict_key
    rttm = rttm[dict_key[0]]

    hypothesis_spkr_tree = IntervalTree(Interval(segment.start, segment.end, label)
                                        for segment, _, label in rttm.itertracks(yield_label=True))

    # file_id, '1', str(row._data['speaker']), f'{ts:.2f}', f'{(ts + duration):.2f}', token

    with open(args.output_stm_transcription, 'w') as f:
        for _, channel, start, dur, token, _ in ctm:
            start, dur = float(start), float(dur)
            hyp_speaker = speaker_for_segment(float(start), float(dur), hypothesis_spkr_tree)
            f.write(f'{dict_key[0]} 1 {hyp_speaker} {start:.3f} {(start + dur):.3f} {token}\n')
