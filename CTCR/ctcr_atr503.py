import glob
import librosa
import numpy as np
import os
import pandas as pd
import struct

from scipy import signal
from sklearn.model_selection import train_test_split

fs = 16000


def features_htk(filename, features, sample_period=100000, sample_kind=9):
    """ features to HTK format binary file """
    sample_size, n_samples = features.shape

    sample_size *= 4  # 4 byte

    header = struct.pack(">iihh", n_samples, sample_period, sample_size, sample_kind)  # big endian

    n_features = sample_size // 4

    frame_byte = b""
    for n in range(n_samples):
        frame = features[:, n]
        feature_byte = b""
        for idx in range(n_features):
            feature_byte += struct.pack(">f", frame[idx])
        frame_byte += feature_byte

    bytecode = header + frame_byte

    with open(filename, "wb") as f:
        f.write(bytecode)


def atr503_features(data_file, ad_list, phn_list, fs=16000):
    num_samples = 0

    scp_file = open("./%s_atr503.scp" % data_file, "w")
    mlf_file = open("./%s_atr503.mlf" % data_file, "w")
    mlf_file.write("#!MLF!#\n")
    for i, (aud_file, phn_file) in enumerate(zip(ad_list, phn_list)):
        # waveform
        data = np.fromfile(aud_file, dtype=">i2").astype("int16")

        norm = data / 32767.0  # normalization [-1, 1]
        pre_emp = signal.lfilter([1.0, -0.97], 1, norm)  # pre-emphasize filter

        # phone label
        df = pd.read_table(phn_file, names=["begin", "end", "label"], sep=" ")
        df.iloc[0, 0] = 0
        begin_end = df.loc[:, ["begin", "end"]].to_numpy()

        #
        # MFCC, delta MFCC, and delta2 MFCC
        #
        for hop_length in [512, 256, 128, 64, 32]:
            mfcc = librosa.feature.mfcc(pre_emp, fs, n_mfcc=13, window="hamming", power=2.0, hop_length=hop_length)

            frame_interval = np.array(begin_end / begin_end.max() * mfcc.shape[1], dtype="int")
            if frame_interval.shape[0] == np.count_nonzero(frame_interval[:, 0] - frame_interval[:, 1]):
                df.loc[:, ["begin", "end"]] = frame_interval * 100000
                break
            else:
                continue

        if frame_interval.shape[0] != np.count_nonzero(frame_interval[:, 0] - frame_interval[:, 1]):
            print("Skip %s and %s" % (aud_file, phn_file))
            continue

        delta = librosa.feature.delta(mfcc)  # delta mfcc
        delta2 = librosa.feature.delta(mfcc, order=2)  # delta2 mfcc

        features = np.concatenate((mfcc, delta, delta2), axis=0)  # features x frames

        #
        # save .htk format
        #
        mlf = "{}_atr503/{:0>5d}".format(data_file, i)
        filename = "./{}_atr503/{:0>5d}.htk".format(data_file, i)

        features_htk(filename, features)

        #
        # scp and mlf
        #
        scp_file.write("%s.mfc=%s[0,%d]\n" % (mlf, filename, features.shape[1] - 1))

        mlf_file.write('"%s.lab"\n' % mlf)
        for _, item in df.iterrows():
            mlf_file.write("%d %d %s\n" % (item["begin"], item["end"], item["label"]))
        mlf_file.write(".\n")  # dot is separator

        num_samples += 1
        if num_samples % 1000 == 0:
            print("Now %d samples..." % num_samples)

    scp_file.close()
    mlf_file.close()

    print("\nNumber of samples", num_samples)


if __name__ == "__main__":
    if not os.path.exists("./train_atr503"):
        os.mkdir("./train_atr503")

    if not os.path.exists("./val_atr503"):
        os.mkdir("./val_atr503")

    ad_list = glob.glob("./atr_503/speech/*.ad")
    phn_list = glob.glob("./atr_503/label/monophone/old/*.lab")

    #
    # split train and val
    #
    train_ad_list, val_ad_list, train_phn_list, val_phn_list = train_test_split(ad_list, phn_list, test_size=0.1,
                                                                                random_state=0)

    #
    # make .list file
    #
    label_set = set()
    for phn_file in train_phn_list:
        with open(phn_file) as f:
            text = f.readlines()
        for t in text:
            label_set.add(t[:-1].split(" ")[-1])
    label_list = list(label_set)
    label_list.sort()
    label_list.append("_")  # blank

    with open("./atr503_mapping.list", "w") as list_file:
        for label in label_list[:-1]:
            list_file.write("%s\n" % label)
        list_file.write("%s" % label_list[-1])

    print("Number of labels : %d" % len(label_list))

    #
    # script and model label file
    #
    atr503_features("train", train_ad_list, train_phn_list)
    atr503_features("val", val_ad_list, val_phn_list)
    
