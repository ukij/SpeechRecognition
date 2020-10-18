import cntk as C

num_feature = 39
num_label = 43

blank_id = num_label - 1

minibatch_size = 1
num_samples = 51


def create_reader(scp_path, mlf_path, list_path, is_train):
    return C.io.MinibatchSource([
        C.io.HTKFeatureDeserializer(C.io.StreamDefs(speech_feature=C.io.StreamDef(shape=num_feature, scp=scp_path))),
        C.io.HTKMLFDeserializer(list_path,
                                C.io.StreamDefs(speech_label=C.io.StreamDef(shape=num_label, mlf=mlf_path)),
                                phoneBoundaries=True)
    ], randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


if __name__ == "__main__":
    #
    # built-in reader
    #
    valid_reader = create_reader("./val_atr503.scp", "./val_atr503.mlf", "./atr503_mapping.list", is_train=False)

    #
    # model, label, and error
    #
    model = C.load_model("./ctcr.model")
    label = C.sequence.input_variable(shape=(num_label,), dtype="float32")

    errs = C.edit_distance_error(model, label, squashInputs=True, tokensToIgnore=[blank_id])

    input_map = {model.arguments[0]: valid_reader.streams.speech_feature, label: valid_reader.streams.speech_label}

    #
    # validation
    #
    sample_count = 0
    error = 0
    while sample_count < num_samples:
        data = valid_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)

        error += errs.eval(data)

        sample_count += data[label].num_sequences

    print("Validation Error {:.2f}%".format(error / num_samples))
    
