import pickle as pk

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from nn_arch import dnn, cnn, rnn

from util import map_path, map_func


batch_size = 32

path_embed = 'feat/embed.pkl'
path_sent = 'feat/sent_train.pkl'
path_label = 'feat/label_train.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

funcs = {'dnn': dnn,
         'cnn': cnn,
         'rnn': rnn}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}


def compile(name, embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input = Input(shape=(seq_len,), dtype='int32')
    embed_input = embed(input)
    func = map_func(name, funcs)
    output = func(embed_input, class_num)
    model = Model(input, output)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def fit(name, epoch, embed_mat, label_inds, sents, labels):
    seq_len = len(sents[0])
    class_num = len(label_inds)
    model = compile(name, embed_mat, seq_len, class_num)
    check_point = ModelCheckpoint(map_path(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit(sents, labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    fit('dnn', 10, embed_mat, label_inds, sents, labels)
    fit('cnn', 20, embed_mat, label_inds, sents, labels)
    fit('rnn', 10, embed_mat, label_inds, sents, labels)
