import pickle as pk

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input, Embedding

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from nn_arch import adnn_core

from util import map_item


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Arial Unicode MS']


def define_core(embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len, name='embed')
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    output = adnn_core(embed_input)
    return Model(input, output)


def load_core(name, embed_mat, seq_len):
    model = define_core(embed_mat, seq_len)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


seq_len = 30

path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

paths = {'adnn': 'model/adnn.h5',
         'crnn': 'model/crnn.h5',
         'rcnn': 'model/rcnn.h5'}

models = {'adnn': load_model(map_item('adnn', paths)),
          'adnn_core': load_core('adnn', embed_mat, seq_len),
          'crnn': load_model(map_item('crnn', paths)),
          'rcnn': load_model(map_item('rcnn', paths))}


def plot_att(text, atts):
    inds = np.arange(len(text))
    plt.bar(inds, atts, width=0.5)
    plt.xlabel('word')
    plt.ylabel('att')
    plt.xticks(inds, text)
    plt.show()


def predict(text, name):
    text = clean(text)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_item(name, models)
    probs = model.predict(pad_seq)[0]
    sort_probs = sorted(probs, reverse=True)
    sort_inds = np.argsort(-probs)
    sort_preds = [ind_labels[ind] for ind in sort_inds]
    formats = list()
    for pred, prob in zip(sort_preds, sort_probs):
        formats.append('{} {:.3f}'.format(pred, prob))
    if name == 'adnn':
        core = map_item(name + '_core', models)
        atts = core.predict(pad_seq)[0]
        plot_att(text, atts[-len(text):])
    return ', '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('adnn: %s' % predict(text, 'adnn'))
        print('crnn: %s' % predict(text, 'crnn'))
        print('rcnn: %s' % predict(text, 'rcnn'))
