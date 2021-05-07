import cntk as C
import librosa
import numpy as np
import pyaudio
import sentencepiece as spm

num_feature = 512
num_word = 8000

UNK = 0
BOS = 1
EOS = 2

MAX = 33

threshold = 20
vad_threshold = 0.03


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


def record(mic_index, time, fs=16000, frames_per_buffer=1024):
    audio = pyaudio.PyAudio()
    data = []
    dt = 1 / fs

    print("Say...")
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, input_device_index=mic_index,
                        frames_per_buffer=frames_per_buffer)
    print("Record.")

    for i in range(int(((time / dt) / fs))):
        frame = stream.read(fs)
        data.append(frame)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    data = b"".join(data)
    data = np.frombuffer(data, dtype="int16")

    return data


if __name__ == "__main__":
    #
    # sentence piece
    #
    spm_model = SentencePiece("./corpus.model")

    model = C.load_model("./sttt.model")

    #
    # speech to text
    #
    while True:
        data = record(1, time=5) / (2 ** 16 / 2 - 1)

        if np.sum(np.where(np.abs(data[20000:]) > vad_threshold)) == 0:
            print("\nI couldn't detect voice activity...\n")
            continue

        #
        # preprocessing
        #
        data, _ = librosa.effects.trim(data, top_db=threshold)  # remove no sound zone
        data /= np.abs(data).max()  # normalization

        data = np.ascontiguousarray(data.reshape(-1, 1), dtype="float32")

        #
        # convolution and transformer
        #
        text = np.identity(num_word, dtype="float32")[BOS].reshape(1, -1)
        dummy = np.identity(num_word, dtype="float32")[EOS].reshape(1, -1)

        for _ in range(MAX):
            prob = model.eval({model.arguments[1]: data, model.arguments[0]: np.vstack((text, dummy))})[0]
            pred = np.identity(num_word, dtype="float32")[prob.argmax(axis=1)[-1]].reshape(1, -1)
            text = np.concatenate((text, pred), axis=0)
            if prob.argmax(axis=1)[-1] == EOS:
                break

        print(">>", spm_model.decode([int(i) for i in text.argmax(axis=1)]))
        
