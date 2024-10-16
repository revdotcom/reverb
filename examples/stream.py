import argparse
import tempfile
import threading
from pydub import AudioSegment
from pydub.playback import play
from wenet import load_model


mdl = load_model('reverb_asr_v1')


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple example of how to run Reverb while streaming an audio file."
    )
    parser.add_argument("audio_file", help="Audio to stream")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Fixed size chunk to cut the audio in milliseconds.",
    )
    return parser.parse_args()


def stream_audio_chunks(file_path, chunk_size_ms=10000):
    audio = AudioSegment.from_file(file_path)

    for i in range(0, len(audio), chunk_size_ms):
        chunk = audio[i:i + chunk_size_ms]
        yield chunk

if __name__ == '__main__':
    args = get_args()
    # Example usage
    for chunk in stream_audio_chunks(args.audio_file, chunk_size_ms=args.chunk_size):
        # Process the chunk
        with tempfile.NamedTemporaryFile() as tfile:
            def play_chunk():
                play(chunk)
            def transcribe_chunk():
                chunk.export(tfile.name, format="wav")
                print(mdl.transcribe(tfile.name))

            thread1 = threading.Thread(target=play_chunk)
            thread2 = threading.Thread(target=transcribe_chunk)

            thread1.start()
            thread2.start()

            # Wait for threads to finish
            thread1.join()
            thread2.join()
