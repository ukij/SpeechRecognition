import librosa
import numpy as np
import os
import pandas as pd
import sentencepiece as spm
import subprocess

from itertools import zip_longest

data_file = "validated"

num_word = 8000

UNK = 0
BOS = 1
EOS = 2


class SentencePiece:
    def __init__(self, spm_path):
        self.model = spm.SentencePieceProcessor()
        self.model.Load(spm_path)

    def encode(self, string):
        return self.model.EncodeAsIds(string)

    def encode_pieces(self, string):
        return self.model.encode_as_pieces(string)

    def decode(self, ids):
        return self.model.DecodeIds(ids)

    def decode_pieces(self, pieces):
        return self.model.decode_pieces(pieces)


def sttt_mp3wav():
    df = pd.read_table("./common_voice/%s.tsv" % data_file, sep="\t")

    if not os.path.exists("./common_voice/%s" % data_file):
        os.mkdir("./common_voice/%s" % data_file)

    mp3_list = df["path"].to_list()
    for mp3_file in mp3_list:
        filename = mp3_file.split(".")[0]
        cmd = "ffmpeg -i ./common_voice/clips/%s.mp3 ./common_voice/%s/%s.wav" % (filename, data_file, filename)

        subprocess.call(cmd, shell=True)


def sttt_sentencepiece():
    df = pd.read_table("./common_voice/%s.tsv" % data_file, sep="\t")
    corpus = df["sentence"].to_list()

    with open("./corpus.txt", "w", encoding="utf-8") as f:
        for sent in corpus:
            f.write("%s\n" % sent)

    spm.SentencePieceTrainer.train(input="./corpus.txt", model_prefix="corpus", vocab_size=num_word)


def sttt_speech2text(threshold):
    #
    # sentence piece
    #
    spm_model = SentencePiece("./corpus.model")

    df = pd.read_table("./common_voice/%s.tsv" % data_file, sep="\t")

    train = df[["path", "sentence"]].to_numpy()

    num_samples = 0
    max_seq_len = 0

    with open("./train_sttt_map.txt", "w") as map_file:
        for i, (file, text) in enumerate(train):
            #
            # word2id
            #
            ids = spm_model.encode(text)
            ids.insert(0, BOS)
            ids.append(EOS)

            if len(ids) > max_seq_len:
                max_seq_len = len(ids)

            #
            # features
            #
            filename = file.split(".")[0]

            data, fs = librosa.load("./common_voice/%s/%s.wav" % (data_file, filename))

            data /= np.abs(data).max()  # normalization
            wave = data[np.where(np.abs(data) > threshold)]  # remove no sound zone

            for (idx, value) in zip_longest(ids, wave, fillvalue=""):
                value_str = " ".join(np.ascontiguousarray(value, dtype="float32").astype(str))
                if idx == "":
                    map_file.write("{} |speech {}\n".format(i, value_str))
                else:
                    map_file.write("{} |text {}:1\t|speech {}\n".format(i, idx, value_str))

            num_samples += 1
            if num_samples % 1000 == 0:
                print("Now %d samples..." % num_samples)

    print("\nNumber of samples", num_samples)
    print("\nMaximum Sequence Length", max_seq_len)


if __name__ == "__main__":
    sttt_mp3wav()

    sttt_sentencepiece()

    sttt_speech2text(threshold=0.01)
    
