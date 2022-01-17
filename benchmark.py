import argparse

import editdistance

from dataset import *
from engine import *
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_type", type=str, required=True)
    args = parser.parse_args()

    dataset = Dataset.create("SpeechAccentArchive")
    print("loaded %s with %.2f hours of data" % (str(dataset), dataset.size_hours()))

    engine = ASREngine.create(ASREngines[args.engine_type])
    print("created %s engine" % str(engine))

    word_error_count = 0
    word_count = 0
    for i in tqdm(range(dataset.size())):
        path, ref_transcript = dataset.get(i)

        transcript = engine.transcribe(path)

        if transcript is None:
            continue

        ref_words = ref_transcript.strip("\n ").lower().split()
        words = transcript.strip("\n ").lower().split()

        word_error_count += editdistance.eval(ref_words, words)
        word_count += len(ref_words)

    print("word error rate : %.2f" % (100 * float(word_error_count) / word_count))
