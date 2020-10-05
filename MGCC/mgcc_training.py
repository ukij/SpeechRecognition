import cntk as C
import numpy as np
import os
import pandas as pd

from cntk.layers import BatchNormalization, Convolution2D, Dense, Dropout, MaxPooling
from cntkx.learners import CyclicalLearningRate

img_channel = 1
img_height = 128
img_width = 128
num_classes = 10

epoch_size = 50
minibatch_size = 32
num_samples = 800

step_size = num_samples // minibatch_size * 10
weight_decay = 0.0005


def mgcc(h):
    with C.layers.default_options(init=C.he_normal(), pad=True, strides=1, bias=False,
                                  map_rank=1, use_cntk_engine=True):
        h = Convolution2D((3, 3), 64)(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = MaxPooling((3, 3), strides=2)(h)

        h = Convolution2D((3, 3), 128)(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = MaxPooling((3, 3), strides=2)(h)

        h = Convolution2D((3, 3), 256)(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = Dense(512, activation=None, init=C.glorot_uniform(), bias=True)(h)
        h = Dropout(dropout_rate=0.5)(h)
        h = Dense(512, activation=None, init=C.glorot_uniform(), bias=True)(h)
        h = Dropout(dropout_rate=0.5)(h)
        h = Dense(num_classes, activation=None, init=C.glorot_uniform(), bias=True)(h)

        return h


if __name__ == "__main__":
    #
    # load dataset
    #
    train_audio, train_label = np.load("./dataset/train_audio.npy"), np.load("./dataset/train_label.npy")

    #
    # input, label and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    label = C.input_variable(shape=(num_classes,), dtype="float32")

    model = mgcc(input)

    #
    # loss function and error metrics
    #
    loss = C.cross_entropy_with_softmax(model, label)
    errs = C.classification_error(model, label)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.momentum_sgd(model.parameters, lr=0.1, momentum=0.9, l2_regularization_weight=weight_decay)
    clr = CyclicalLearningRate(learner, base_lr=1e-5, max_lr=1e-3, ramp_up_step_size=step_size,
                               minibatch_size=minibatch_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, errs), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)

    #
    # training
    #
    logging = {"epoch": [], "loss": [], "error": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_loss = 0
        epoch_metric = 0
        while sample_count < num_samples:
            data = {input: train_audio[sample_count: sample_count + minibatch_size],
                    label: train_label[sample_count: sample_count + minibatch_size]}

            trainer.train_minibatch(data)

            clr.batch_step()

            sample_count += minibatch_size
            epoch_loss += trainer.previous_minibatch_loss_average
            epoch_metric += trainer.previous_minibatch_evaluation_average

        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / (num_samples / minibatch_size))
        logging["error"].append(epoch_metric / (num_samples / minibatch_size))

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    model.save("./gtzan.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./gtzan.csv", index=False)
    print("Saved logging.")
    
