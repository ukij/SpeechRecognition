import cntk as C
import librosa
import numpy as np
import pyaudio

from scipy import signal

num_label = 43

blank_id = num_label - 1


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


def play(data, fs=16000):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(2), channels=1, rate=fs, output=True)
    stream.write(data.tostring())
    stream.close()
    audio.terminate()


def ctcr_features(data, fs):
    norm = data / 32767.0  # normalization [-1, 1]
    pre_emp = signal.lfilter([1.0, -0.97], 1, norm)  # pre-emphasize filter

    #
    # MFCC, delta MFCC, and delta2 MFCC
    #
    mfcc = librosa.feature.mfcc(pre_emp, fs, n_mfcc=13, window="hamming", power=2.0, hop_length=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.concatenate((mfcc, delta, delta2), axis=0).transpose(1, 0)  # features x frames -> frames x features


if __name__ == "__main__":
    #
    # label list and model
    #
    with open("./atr503_mapping.list") as f:
        id2label = f.read().split("\n")

    model = C.load_model("./ctcr.model")

    #
    # phoneme prediction
    #
    data = record(1, time=3)
    play(data)
    feature = ctcr_features(data, fs=16000)

    pred = model.eval({model.arguments[0]: np.ascontiguousarray(feature, dtype="float32")})[0]

    pred_label = pred.argmax(axis=1)
    pred_label = pred_label[np.append(pred_label[:-1] != pred_label[1:], True)]  # remove redundancy
    pred_label = pred_label[pred_label != blank_id]  # remove blank

    print([id2label[int(p)] for p in pred_label])
    
