# Evaluating ASR Systems
We provide two python scripts to facilitate aligment and metric aggregation:
1. `scoring_commands.py` prints to STDOUT a series of commands that can be run to get the scoring between a reference and hypothesis transcript. _How_ to run these commands is up to the user, but we highly recommend using parallelization when possible. This script assumes a dataset from our [speech-datasets](https://github.com/revdotcom/speech-datasets) repository is being used; these have reference "nlp" transcripts along with reference normalizations that can improve the quality of scoring. This script also depends on having the [fstalign](https://github.com/revdotcom/fstalign/tree/develop) binary locally, follow the instructions on that repository to install. For more information on how to run this script, run `python3 scoring_commands.py --help`.
2. `aggregate_scoring.py` aggregates the output results from `scoring_commands.py` to produce the key metrics we use to evaluate ASR systems. For more information on how to run this script, run `python3 aggregate_scoring.py --help`.

## Example
```bash
~$ FSTALIGN_BINARY=fstalign/fstalign
~$ NLP_REFERENCE_DIRECTORY=speech-datasets/earnings21/transcripts/nlp_references/
~$ ASR_HYPOTHESIS_DIRECTORY=asr_output
~$ FSTALIGN_OUTPUT_DIRECTORY=scoring_output
~$ NLP_NORMALIZATIONS_DIRECTORY=speech-datasets/earnings21/transcripts/normalizations/
~$ FSTALIGN_SYNONYMS_FILE=fstalign/sample_data/synonyms.rules.txt

~$ python3 scoring_commands.py \
${FSTALIGN_BINARY} \
${NLP_REFERENCE_DIRECTORY} \
${ASR_HYPOTHESIS_DIRECTORY} \
${FSTALIGN_OUTPUT_DIRECTORY} \
--ref-norm ${NLP_NORMALIZATIONS_DIRECTORY} \
--synonyms-file ${FSTALIGN_SYNONYMS_FILE} > cmds.sh

~$ head -n 1 cmds.sh
fstalign/fstalign wer --ref speech-datasets/earnings21/transcripts/nlp_references/4387865.nlp --hyp asr_output/4387865.ctm --json-log scoring_output/4387865.json --ref-json speech-datasets/earnings21/transcripts/normalizations/4387865.norm.json --syn fstalign/sample_data/synonyms.rules.txt

~$  ./cmds.sh
~$  python3 aggregate_scoring.py ${FSTALIGN_OUTPUT_DIRECTORY}
TOTAL WER:      29172/374486 = 7.79%
Insertion Rate: 6443/374486 = 1.72%
Deletion Rate:  8405/374486 = 2.24%
Substitution Rate:      14324/374486 = 3.82%
```
