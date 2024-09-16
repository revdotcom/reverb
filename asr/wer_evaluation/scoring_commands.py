#!/usr/bin/env python
# -*- coding: utf-8 -*-


from argparse import ArgumentParser
from pathlib import Path
from typing import Optional


def init_args():
    parser = ArgumentParser(
        description="Generates fstalign a list of commands that"
        " will perform adequate alignment between a test-suite"
        " and hypothesis directory. This script assumes the"
        " hypotheses are in CTM format and the references are"
        " in NLP format."
    )
    parser.add_argument(
        'fstalign', type=Path,
        help="Path to the fstalign binary."
    )
    parser.add_argument(
        'ref', type=Path,
        help="Path to test suite transcript file(s). Pass in either"
        " one file or a directory. Files *must* be in NLP format."
    )
    parser.add_argument(
        'hyp', type=Path,
        help="Path to ASR Hypothesis file(s). Pass in either one file"
        " or a directory. *Assumes files are in CTM format*."
    )
    parser.add_argument(
        'out', type=Path,
        help="Path to output directory to contain fstalign files."
    )
    parser.add_argument(
        '--ref-norm', type=Path,
        default=None,
        help="Path to test suite normalization file(s). Pass in either"
        " one file or a directory. *In order for fstalign to use"
        " normalizations files, references in NLP format with normalization"
        " tags are required. Unexpected results may occur if used"
        " incorrectly.*"
    )
    parser.add_argument(
        '--synonyms-file', type=Path,
        default=None,
        help="Path to fstalign, synonym file."
    )
    return parser.parse_args()


def prepare_IO(
    ref_path: Path,
    hyp_path: Path,
    out_path: Path,
    ref_norm_path: Optional[Path]=None,
):
    """Determines if the hyp_path provided is a directory or a file
    and appropriately identifies paths for fstalign files. Creates
    a Generator to return the files.

    out_path will always be made into a directory

    When hyp_path is a file:
    * ref_path and ref_norm_path are assumed to be files.
    * The resulting JSON from fstalign will be in out_path with the
      same name as the hyp_path.

    When hyp_path is a directory:
    * ref_path and ref_norm_path are assumed to be directories.
      For each hypothesis CTM in hyp_path, the equivalent reference
      transcripts and normalizations are assumed to have the same
      name as the hypothesis file.
    * The resulting JSONs from fstalign will be in out_path with the
      same name as the CTMs in hyp_path.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    if hyp_path.is_dir():
        for hyp_file in hyp_path.glob("**/*.ctm"):
            hyp_name = hyp_file.stem
            ref_file = (ref_path / (hyp_name + ".nlp")).resolve()
            out_file = (out_path / (hyp_name + ".log.json")).resolve()
            ref_norm_file = None
            if ref_norm_path:
                ref_norm_file = (ref_norm_path / (hyp_name + '.norm.json')).resolve()
            yield ref_file, hyp_file.resolve(), out_file, ref_norm_file
    else:
        out_file = (out_path / (hyp_path.stem + ".log.json")).resolve()
        if ref_norm_path:
            ref_norm_path = ref_norm_path.resolve()
        yield ref_path.resolve(), hyp_path.resolve(), out_file, ref_norm_path


if __name__ == '__main__':
    args = init_args()

    for ref_file, hyp_file, out_file, ref_norm_file in prepare_IO(args.ref, args.hyp, args.out, args.ref_norm):
        alignment_command = [
            str(args.fstalign),
            "wer",
            "--ref",
            str(ref_file),
            "--hyp",
            str(hyp_file),
            "--json-log",
            str(out_file),
        ]
        if ref_norm_file:
            alignment_command.extend([
                "--ref-json",
                str(ref_norm_file),
            ])
        if args.synonyms_file:
            alignment_command.extend([
                "--syn",
                str(args.synonyms_file),
            ])

        print(' '.join(alignment_command))
