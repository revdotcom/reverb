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
from pyannote.core import Annotation

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

    """ We had to hack rttm format a little bit so it can store both information from ctm file (token and 
        token confidence). This is not implemented in pyannote, so we wrote it ourselves.
        For rttm description see https://github.com/nryant/dscore?tab=readme-ov-file#rttm
        We are using `Orthography Field` for token and `Confidence Score` for token confidence.
        Trying to read this rttm format in pyannote might yield unexpected results or even errors.
    """
    parser.add_argument('output', help='output file in rttm format as described above')

    args = parser.parse_args()

    ctm = read_ctm(args.ctm_transcription)
    rttm = load_rttm(args.diarization_rttm)
    dict_key = list(rttm.keys())
    assert len(dict_key) == 1, dict_key
    rttm = rttm[dict_key[0]]

    hypothesis_spkr_tree = IntervalTree(Interval(segment.start, segment.end, label)
                                        for segment, _, label in rttm.itertracks(yield_label=True))

    with open(args.output, 'w') as f:
        for utt_id, channel, start, dur, token, confidence in ctm:
            start, dur, confidence = float(start), float(dur), float(confidence)
            hyp_speaker = speaker_for_segment(float(start), float(dur), hypothesis_spkr_tree)
            f.write(f"SPEAKER {dict_key[0]} 1 {start:.3f} {dur:.3f} {token} <NA> {hyp_speaker} {confidence:.2f} <NA>\n")
