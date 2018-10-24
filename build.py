import pickle as pk

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from nn_arch import adnn, crnn, rcnn

from util import map_item


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

funcs = {'adnn': adnn,
         'crnn': crnn,
         'rcnn': rcnn}

paths = {'adnn': 'model/adnn.h5',
         'crnn': 'model/crnn.h5',
         'rcnn': 'model/rcnn.h5',
         'adnn_plot': 'model/plot/dnn.png',
         'crnn_plot': 'model/plot/cnn.png',
         'rcnn_plot': 'model/plot/rnn.png'}


def compile(name, embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                       weights=[embed_mat], input_length=seq_len, trainable=True)
    # embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
    #                   weights=[embed_mat], input_length=seq_len, trainable=False)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    output = func(embed_input, class_num)
    model = Model(input, output)
    model.summary()
    plot_model(model, map_item(name + '_plot', paths), show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def fit(name, epoch, embed_mat, label_inds, sents, labels):
    seq_len = len(sents[0])
    class_num = len(label_inds)
    model = compile(name, embed_mat, seq_len, class_num)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit(sents, labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    fit('adnn', 10, embed_mat, label_inds, sents, labels)
    fit('crnn', 10, embed_mat, label_inds, sents, labels)
    fit('rcnn', 10, embed_mat, label_inds, sents, labels)
