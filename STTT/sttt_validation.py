import cntk as C
import nltk
import numpy as np
import sentencepiece as spm

from nltk.translate.bleu_score import sentence_bleu

num_word = 8000

UNK = 0
BOS = 1
EOS = 2

MAX = 33


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


def sttt_bleu(num_samples):
    #
    # sentence piece
    #
    spm_model = SentencePiece("./corpus.model")

    #
    # built-in reader
    #
    dev_reader = create_reader("./val_sttt_map.txt", is_train=False)

    #
    # model
    #
    model = C.load_model("./sttt.model")
    targets = model.arguments[0] * 1

    input_map = {model.arguments[1]: dev_reader.streams.speech, model.arguments[0]: dev_reader.streams.text}

    def id2word(ids, spm_model):
        words = []
        for w in ids:
            if w == 1:
                words.append("<BOS>")
            elif w == 2:
                words.append("<EOS>")
            else:
                words.append(spm_model.decode([int(w)]))
        return words

    #
    # bilingual evaluation understudy
    #
    method = nltk.translate.bleu_score.SmoothingFunction()
    bleu4 = []
    bleu1 = []
    for i in range(num_samples):
        data = dev_reader.next_minibatch(1, input_map=input_map)

        text = np.identity(num_word, dtype="float32")[BOS].reshape(1, -1)
        dummy = np.identity(num_word, dtype="float32")[EOS].reshape(1, -1)

        target = targets.eval({targets.arguments[0]: data[model.arguments[0]].data})[0]

        for _ in range(MAX):
            try:
                prob = model.eval({model.arguments[1]: data[model.arguments[1]].data,
                                   model.arguments[0]: np.vstack((text, dummy))})[0]
                pred = np.identity(num_word, dtype="float32")[prob.argmax(axis=1)[-1]].reshape(1, -1)
                text = np.concatenate((text, pred), axis=0)
                if prob.argmax(axis=1)[-1] == EOS:
                    break
            except RuntimeError:
                break

        reference = id2word(target.argmax(axis=1), spm_model)
        candidate = id2word(text[:-1, :].argmax(axis=1), spm_model)

        bleu4.append(sentence_bleu(reference[1:-1], candidate[1:], smoothing_function=method.method3))
        bleu1.append(sentence_bleu(reference[1:-1], candidate[1:], weights=(1,), smoothing_function=method.method3))

    bleu4_score = np.array(bleu4)
    bleu1_score = np.array(bleu1)

    print("BLEU-4 Score {:.2f}".format(bleu4_score.mean() * 100))
    print("BLEU-1 Score {:.2f}".format(bleu1_score.mean() * 100))


if __name__ == "__main__":
    sttt_bleu(num_samples=1219)
    
