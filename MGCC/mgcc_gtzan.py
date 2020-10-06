import glob
import librosa
import numpy as np
import os

from sklearn.model_selection import train_test_split

path = "./gtzan"

img_channel = 1
img_height = 128
img_width = 128

num_classes = 10


if __name__ == "__main__":
    categories = os.listdir(path)

    wav_list = []
    for category in categories:
        file_list = glob.glob("{}/{}/*.wav".format(path, category))
        wav_list += file_list

    label_dict = {c: i for i, c in enumerate(categories)}

    #
    # log-mel spectrogram and label
    #
    audio_list, label_list = [], []
    for file in wav_list:
        label = label_dict[file.rsplit("\\")[-1].split(".", 1)[0]]

        data, fs = librosa.load(file)

        log_mel = np.log10(librosa.feature.melspectrogram(data, fs, window="hann") + 1.0)
        
        for i in range(len(log_mel) // img_width):
            audio_list.append(log_mel[:, i * img_width:(i + 1) * img_width])
            label_list.append(label)
    
    gtzan_audio = np.array(audio_list, dtype="float32").reshape(-1, img_channel, img_height, img_width)
    gtzan_label = np.identity(num_classes)[label_list].astype("float32")

    #
    # split train and test
    #
    train_audio, test_audio, train_label, test_label = train_test_split(gtzan_audio, gtzan_label, test_size=0.2)

    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")

    np.save("./dataset/train_audio.npy", train_audio)
    np.save("./dataset/train_label.npy", train_label)
    np.save("./dataset/test_audio.npy", test_audio)
    np.save("./dataset/test_label.npy", test_label)
    
