#!/usr/bin/env python
# -*- coding: utf-8 -*-


from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict


def init_args():
    parser = ArgumentParser(
        description="Takes in directory of fstalign outputs and calculates "
        " the aggregate WER metric over the full test suite."
    )
    parser.add_argument(
        'fstalign_out', type=Path,
        help="Path to an fstalign alignment output directory. This script"
        " relies specifically on the output from setting the --log-json"
        " flag in fstalign."
    )
    return parser.parse_args()


@dataclass
class WERAggregator:
    insertion_count: int = 0
    deletion_count: int = 0
    substitution_count: int = 0
    correct_count: int = 0
    reference_count: int = 0

    def update(self, alignment_dict: Dict[str, float]):
        """Given a dictionary with alignment statistics from fstalign,
        update the corresponding counts.
        """
        self.insertion_count += alignment_dict['insertions']
        self.deletion_count += alignment_dict['deletions']
        self.substitution_count += (alignment_dict['numErrors'] - alignment_dict['insertions'] - alignment_dict['deletions'])
        self.correct_count += (alignment_dict['numWordsInReference'] - alignment_dict['substitutions'] - alignment_dict['deletions'])
        self.reference_count += alignment_dict['numWordsInReference']

    @property
    def num_errors(self):
        """Calculates the total number of errors of the aggregator.
        """
        return self.insertion_count + self.deletion_count + self.substitution_count

    def check_state(self):
        """Ensures all assumptions are valid prior to calculations. Raises
        Raises an error if assumptions are broken.
        """
        if self.reference_count == 0:
            raise RuntimeError("Something went wrong! Cannot compute a rate when `reference_count` is 0.")

    def insertion_rate(self) -> float:
        """Returns a float of the aggregator's insertion rate.
        """
        self.check_state()
        return self.insertion_count / self.reference_count

    def deletion_rate(self) -> float:
        """Returns a float of the aggregator's deletion rate.
        """
        self.check_state()
        return self.deletion_count / self.reference_count

    def substitution_rate(self) -> float:
        """Returns a float of the aggregator's substitution rate.
        """
        self.check_state()
        return self.substitution_count / self.reference_count

    def wer(self) -> float:
        """Returns a float of the aggregator's WER (word error rate).
        """
        self.check_state()
        return self.num_errors / self.reference_count

    def summary(self) -> str:
        """Provides a string summary of all aggregator's state in a formatted string.
        This includes:
         * WER
         * Insertion Rate
         * Deletion Rate
         * Substitution Rate
        """
        def format_rate(title: str, numerator: int, rate: float) -> str:
            """Creates a string to represent a rate given its numerator
            and denominator.
            """
            return f"{title}:\t{numerator}/{self.reference_count} = {rate:3.2%}"

        summary = [
            format_rate("TOTAL WER", self.num_errors, self.wer()),
            format_rate("Insertion Rate", self.insertion_count, self.insertion_rate()),
            format_rate("Deletion Rate", self.deletion_count, self.deletion_rate()),
            format_rate("Substitution Rate", self.substitution_count, self.substitution_rate()),
        ]
        return '\n'.join(summary)


if __name__ == '__main__':
    args = init_args()

    aggregator = WERAggregator()

    for json_path in args.fstalign_out.glob("*.json"):
        with json_path.open('r') as jfile:
            alignment_results = json.load(jfile)
            aggregator.update(alignment_results['wer']['bestWER'])

    print(aggregator.summary())
