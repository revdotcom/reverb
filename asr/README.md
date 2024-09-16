## Getting Started

### Retrieve the model
The Reginald ASR model v1 is stored on HuggingFace https://huggingface.co/Revai/reginald.

### Install inference requirements
```
pip install -r requirements.txt
export PYTHONPATH="$(pwd)":$PYTHONPATH
```

### Example usage
do python wenet/bin/recognize_wav.py --config --checkpoint --audio ....
or try our demo here: ____


## About
Rev’s Reginald ASR model was trained on 200,000 hours of English speech, all expertly transcribed by humans - the largest corpus of human transcribed audio ever used to train an open-source model. The quality of this data has produced the world’s most accurate English automatic speech recognition (ASR) system, using an efficient model architecture that can be run on either CPU or GPU. Additionally, Reginald provides user control over the level of verbatimicity of the output transcript, making it ideal for both clean, readable transcription and use-cases like audio editing that require transcription of every spoken word including hesitations and re-wordings. Users can specify fully verbatim, fully non-verbatim, or anywhere in between for their transcription output.

## Transcription Style Options
Reginald was trained to produce transcriptions in either a verbatim style, in which every word is transcribed as spoken; or a non-verbatim style, in which disfluencies may be removed from the transcript. Users can specify Reginald's output style with the clean_read parameter. 0 corresponds to a verbatim transcript and 1 corresponds to a non-verbatim transcript.

See our demo _____

## Usage
Example usage: python wenet/bin/recognize_wav.py --clean_read 0 --config .. --checkpoint .. --audio ..

## Benchmarking
See wer_evaluation folder for details and results.

