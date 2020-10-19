import cntk as C
import os
import pandas as pd

from cntk.layers import Dense, Dropout, LayerNormalization, LSTM, Recurrence
from cntkx.learners import CyclicalLearningRate

num_feature = 39
num_label = 43

num_hidden = 512
num_stack = 3

blank_id = num_label - 1

epoch_size = 100
minibatch_size = 1024
num_samples = 452

sample_size = 16
step_size = num_samples // sample_size * 10


def create_reader(scp_path, mlf_path, list_path, is_train):
    return C.io.MinibatchSource([
        C.io.HTKFeatureDeserializer(C.io.StreamDefs(speech_feature=C.io.StreamDef(shape=num_feature, scp=scp_path))),
        C.io.HTKMLFDeserializer(list_path,
                                C.io.StreamDefs(speech_label=C.io.StreamDef(shape=num_label, mlf=mlf_path)),
                                phoneBoundaries=True)
    ], randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def ctcr(h):
    with C.layers.default_options(enable_self_stabilization=True):
        for i in range(num_stack):
            if i == 0:
                h_forward = LayerNormalization()(Recurrence(LSTM(num_hidden // 2))(h))
                h_backward = LayerNormalization()(Recurrence(LSTM(num_hidden // 2), go_backwards=True)(h))
                h = Dropout(dropout_rate=0.1)(C.splice(h_forward, h_backward))
            else:
                h_forward = LayerNormalization()(Recurrence(LSTM(num_hidden // 2))(h))
                h_backward = LayerNormalization()(Recurrence(LSTM(num_hidden // 2), go_backwards=True)(h))
                h = Dropout(dropout_rate=0.1)(C.splice(h_forward, h_backward)) + h

        h = Dense(num_label)(h)

        return h


def connectionist_temporal_classification_loss(output_vector, target_vector, blank_id, delay_const=-1):
    return C.forward_backward(
        C.labels_to_graph(target_vector), output_vector, blankTokenId=blank_id, delayConstraint=delay_const)


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train_atr503.scp", "./train_atr503.mlf", "./atr503_mapping.list", is_train=True)

    #
    # feature, label and model
    #
    feature = C.sequence.input_variable(shape=(num_feature,), needs_gradient=True, dtype="float32")
    label = C.sequence.input_variable(shape=(num_label,), dtype="float32")

    model = ctcr(feature)

    input_map = {feature: train_reader.streams.speech_feature, label: train_reader.streams.speech_label}

    #
    # loss function and error metrics
    #
    loss = connectionist_temporal_classification_loss(model, label, blank_id, delay_const=3)
    errs = C.edit_distance_error(model, label, squashInputs=True, tokensToIgnore=[blank_id])

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(model.parameters, lr=C.learning_parameter_schedule_per_sample(0.1), momentum=0.9,
                     gradient_clipping_threshold_per_sample=sample_size, gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lr=1e-5, max_lr=1e-3,
                               ramp_up_step_size=step_size, minibatch_size=sample_size)
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
            data = train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)

            trainer.train_minibatch(data)

            clr.batch_step()

            minibatch_count = data[feature].num_sequences
            sample_count += minibatch_count
            epoch_loss += trainer.previous_minibatch_loss_average * minibatch_count
            epoch_metric += trainer.previous_minibatch_evaluation_average * minibatch_count

        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / num_samples)
        logging["error"].append(epoch_metric / num_samples)

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    model.save("./ctcr.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./ctcr.csv", index=False)
    print("Saved logging.")
    
