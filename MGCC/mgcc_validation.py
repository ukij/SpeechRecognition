import cntk as C
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import confusion_matrix

plt.rcParams["font.size"] = "12"

num_classes = 10

minibatch_size = 32
num_samples = 200


if __name__ == "__main__":
    #
    # load dataset
    #
    test_audio, test_label = np.load("./dataset/test_audio.npy"), np.load("./dataset/test_label.npy")

    #
    # model and label
    #
    model = C.load_model("./gtzan.model")
    label = C.input_variable(shape=num_classes, dtype="float32")

    errs = C.classification_error(model, label)

    #
    # validation
    #
    sample_count = 0
    error_count = 0
    pred_list, true_list = [], []
    while sample_count < num_samples:
        data = {model.arguments[0]: test_audio[sample_count: sample_count + minibatch_size],
                label: test_label[sample_count: sample_count + minibatch_size]}

        error_count += errs.eval(data).sum()
        pred_list += list(C.softmax(model).eval({model.arguments[0]: data[model.arguments[0]]}).argmax(axis=1))
        true_list += list(data[label].argmax(axis=1))

        sample_count += minibatch_size

    print("Validation Accuracy {:.2f}%".format((num_samples - error_count) / num_samples * 100))

    #
    # confusion matrix
    #
    categories = os.listdir("./gtzan")
    confusion = confusion_matrix(true_list, pred_list)

    fig, ax = plt.subplots()
    im = ax.imshow(confusion, cmap="hot")

    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    im = ax.imshow(confusion, cmap="hot")

    plt.show()
    
