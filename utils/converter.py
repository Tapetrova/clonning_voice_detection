import os
from glob import glob

from pydub import AudioSegment


def convert(input_path: str, output_path: str):
    audio = AudioSegment.from_file(input_path, format="m4a")

    # Export the file as .wav
    audio.export(output_path, format="wav")


if __name__ == "__main__":
    folder2convert = glob("audio_samples/samples2convert/*")
    for input_path in folder2convert:
        convert(input_path, output_path=os.path.join("audio_samples/samples_converted",
                                                     f"{os.path.splitext(os.path.basename(input_path))[0]}.wav"))



