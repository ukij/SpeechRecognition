import cntk as C
import cntkx as Cx
import os
import pandas as pd
import sentencepiece as spm

from cntk.layers import BatchNormalization, Dropout, LayerNormalization, SequentialConvolution
from cntk.layers.blocks import _inject_name
from cntkx.layers import Embedding
from cntkx.learners import CyclicalLearningRate

num_head = 8
num_hidden = 512
num_stack = 6
num_word = 8000

iteration = 50000
max_seq_len = 33
minibatch_size = 655360
num_samples = 6158

sample_size = 16
step_size = num_samples // sample_size * 10


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


def create_reader(path, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        speech=C.io.StreamDef(field="speech", shape=1, is_sparse=False),
        text=C.io.StreamDef(field="text", shape=num_word, is_sparse=True))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)

  
def PositionalEncoding(num_hidden, name=''):
    """ Positional Encoding """
    @C.BlockFunction('PositionalEncoding', name)
    def positional_encoding(x):
        angle_rates = C.constant(1 / np.power(10000, (2 * (np.arange(num_hidden) // 2)) / num_hidden), dtype="float32")
        angle_rates_pi = C.constant([0, np.pi / 2] * (num_hidden // 2), dtype="float32")

        pos = Cx.sequence.position(x)  # pos: [#, *] [1, ]

        pe = C.sin(pos * angle_rates + angle_rates_pi)  # cos(x) = sin(x + pi/2)
        return pe

    return positional_encoding


def ScaledDotProductAttention(obey_sequence_order=None, max_seq_len=None, name=''):
    """ Scaled Dot Product Attention """
    @C.Function
    def attention(query, key, value):
        dk = C.sqrt(C.reduce_sum(C.ones_like(query)))  # dk: [#, *] [1, ] and value = int(dim_of_query)

        unpacked_key = C.sequence.unpack(key, padding_value=0, no_mask_output=True)  # [#] [-3, key_dim]
        unpacked_value = C.sequence.unpack(value, padding_value=0, no_mask_output=True)  # [#] [-3, value_dim]

        broadcasted_key = C.sequence.broadcast_as(unpacked_key, query)  # [#, *] [-3, key_dim]
        scaled = C.times_transpose(query, broadcasted_key) / dk

        if obey_sequence_order and max_seq_len:
            unpacked_scaled, scaled_mask = C.sequence.unpack(scaled, padding_value=0).outputs

            minus_inf = C.constant(-1e+30)
            valid_connections = C.Constant(np.tril(np.ones((max_seq_len, max_seq_len)), k=0))  # [] [max_seq, max_seq]
            valid_connections = C.reconcile_dynamic_axes(valid_connections, unpacked_scaled)  # [#] [max_seq, max_seq]
            valid_connections = C.crop_manual(valid_connections, unpacked_scaled, 0, 0)  # [#] [-3, -3]
            unpacked_scaled = C.element_select(valid_connections, unpacked_scaled, minus_inf)  # [#] [-3, -3]
            scaled = C.to_sequence_like(unpacked_scaled, query)  # [#, *] [-3]

        elif obey_sequence_order and not max_seq_len:
            raise ValueError("max_seq_len must be defined when obey_sequence_order is True")

        attention_weights = C.softmax(scaled, axis=-1)
        attention_weights = C.layers.Label('attention_weights')(attention_weights)

        attended = C.times(attention_weights, C.sequence.broadcast_as(unpacked_value, query))  # [#, *] [value_dim,]
        return attended

    return _inject_name(attention, name)


def MultiHeadAttention(num_heads, model_dim, key_init=C.glorot_uniform(), query_init=C.glorot_uniform(),
                       value_init=C.glorot_uniform(), concat_init=C.glorot_uniform(),
                       obey_sequence_order=None, max_seq_len=None, name=''):
    """ Multi Head Attention """
    head_dim = model_dim // num_heads

    query_linear = [Dense(head_dim, init=query_init) for _ in range(num_heads)]
    key_linear = [Dense(head_dim, init=key_init) for _ in range(num_heads)]
    value_linear = [Dense(head_dim, init=value_init) for _ in range(num_heads)]
    concat_linear = Dense(model_dim, init=concat_init)

    scaled_dot_product_attention = [ScaledDotProductAttention(
        obey_sequence_order, max_seq_len, name=name + "_%d" % (i + 1)) for i in range(num_heads)]

    @C.Function
    def inner(query, key, value):
        queries = [query_linear[i](C.slice(query, 0, i * head_dim, (i + 1) * head_dim)) for i in range(num_heads)]
        keys = [key_linear[i](C.slice(key, 0, i * head_dim, (i + 1) * head_dim)) for i in range(num_heads)]
        values = [value_linear[i](C.slice(value, 0, i * head_dim, (i + 1) * head_dim)) for i in range(num_heads)]

        attention_outputs = [
            scaled_dot_product_attention[i](q, k, v) for i, (q, k, v) in enumerate(zip(queries, keys, values))]

        return concat_linear(C.splice(*attention_outputs))

    return _inject_name(inner, name)


def BertPositionwiseFeedForward(outer_dim, inner_dim, dropout_rate, name=''):
    inner_dense = Dense(inner_dim, init=C.normal(0.02), name='inner')
    outer_dense = Dense(outer_dim, init=C.normal(0.02), name='outer')
    dropout = Dropout(dropout_rate)

    @C.BlockFunction('BertPositionwiseFeedForward', name)
    def bert_positionwise_feedforward(x):
        return dropout(outer_dense(Cx.gelu_fast(inner_dense(x))))

    return bert_positionwise_feedforward


def sequential_convolution(h):
    with C.layers.default_options(init=C.normal(0.02), pad=True, bias=False, map_rank=1, use_cntk_engine=True):
        h = SequentialConvolution(513, 128, pad=False, strides=64)(h)
        h = BatchNormalization()(h)
        h = Cx.mish(h)

        h = SequentialConvolution(65, 256, strides=8)(h)
        h = BatchNormalization()(h)
        h = Cx.mish(h)

        h = SequentialConvolution(5, num_hidden, strides=2)(h)
        h = BatchNormalization()(h)
        h = Cx.mish(h)

    return h


def sttt(encode, decode):
    spc_embed = _inject_name(sequential_convolution(encode), "features")
    pe = PositionalEncoding(num_hidden)
    txt_embed, dense = Embedding(num_hidden, init=C.normal(0.02), enable_weight_tying=True)

    #
    # encoder
    #
    h_enc = spc_embed(encode)
    h_enc = h_enc + pe(h_enc)

    for i in range(num_stack):
        mha = MultiHeadAttention(num_head, num_hidden, name="enc%d" % (i + 1))(h_enc, h_enc, h_enc)  # self attention
        mha = Dropout(0.1)(mha)

        add_norm = LayerNormalization()(mha + h_enc)

        ffn = BertPositionwiseFeedForward(num_hidden, num_hidden * 4, dropout_rate=0.1)(add_norm)
        h_enc = LayerNormalization()(ffn + add_norm)

    #
    # decoder
    #
    h_dec = txt_embed(decode)
    h_dec = h_dec + pe(h_dec)

    for i in range(num_stack):
        mha = MultiHeadAttention(num_head, num_hidden,  # masked self attention
                                 obey_sequence_order=True, max_seq_len=max_seq_len)(h_dec, h_dec, h_dec)
        mha = Dropout(0.1)(mha)

        h_dec = LayerNormalization()(mha + h_dec)

        mha = MultiHeadAttention(num_head, num_hidden)(h_dec, h_enc, h_enc)  # source-target attention
        mha = Dropout(0.1)(mha)

        add_norm = LayerNormalization()(mha + h_dec)

        ffn = BertPositionwiseFeedForward(num_hidden, num_hidden * 4, dropout_rate=0.1)(add_norm)
        h_dec = LayerNormalization()(ffn + add_norm)

        h_dec = ffn + h_dec

    return dense(h_dec)


def write_text(model, data, encode, decode, spm_model):
    outputs = model.eval({encode: data[encode].data, decode: data[decode].data})

    for output in outputs:
        out = spm_model.decode([int(o) for o in output.argmax(axis=1)])
        print(out)

    return out


if __name__ == "__main__":
    #
    # sentence piece
    #
    spm_model = SentencePiece("./corpus.model")

    #
    # built-in reader
    #
    train_reader = create_reader("./train_sttt_map.txt", is_train=True)

    #
    # speech, text and model
    #
    speech = C.sequence.input_variable(shape=(1,), needs_gradient=True, dtype="float32")
    text = C.sequence.input_variable(shape=(num_word,), dtype="float32", sequence_axis=C.Axis("Text"))

    model = sttt(speech, C.sequence.slice(text, 0, -1))

    input_map = {speech: train_reader.streams.speech, text: train_reader.streams.text}

    #
    # loss function and perplexity
    #
    loss = C.cross_entropy_with_softmax(model, C.sequence.slice(text, 1, 0))
    ppl = C.exp(loss)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(model.parameters, lr=C.learning_parameter_schedule_per_sample(0.1), momentum=0.9,
                     gradient_clipping_threshold_per_sample=sample_size, gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lr=1e-8, max_lr=1e-4,
                               ramp_up_step_size=step_size, minibatch_size=sample_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, ppl), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)
    
    #
    # training
    #
    logging = {"step": [], "loss": [], "ppl": [], "text": []}
    for step in range(iteration):
        data = train_reader.next_minibatch(minibatch_size, input_map=input_map)

        trainer.train_minibatch(data)

        clr.batch_step()

        if step % (iteration // 100) == 0:
            step_loss = trainer.previous_minibatch_loss_average
            step_metric = trainer.previous_minibatch_evaluation_average

            txt = write_text(model, data, speech, text, spm_model)

            #
            # loss and ppl logging
            #
            logging["step"].append(step)
            logging["loss"].append(step_loss)
            logging["ppl"].append(step_metric)
            logging["text"].append(txt)

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    model.save("./sttt.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./sttt.csv", index=False)
    print("Saved logging.")
    
