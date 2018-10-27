import pickle as pk

import re

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input, Embedding

from keras.preprocessing.sequence import pad_sequences

from nn_arch import adnn_plot

from util import load_word_re, load_type_re, load_pair, word_replace, map_item


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Arial Unicode MS']


def define_adnn_plot(embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len, name='embed')
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    output = adnn_plot(embed_input)
    return Model(input, output)


def load_adnn_plot(name, embed_mat, seq_len):
    model = define_adnn_plot(embed_mat, seq_len)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


seq_len = 30

path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = dict()
for label, ind in label_inds.items():
    ind_labels[ind] = label

paths = {'adnn': 'model/adnn.h5',
         'crnn': 'model/crnn.h5',
         'rcnn': 'model/rcnn.h5'}

models = {'adnn': load_model(map_item('adnn', paths)),
          'adnn_plot': load_adnn_plot('adnn', embed_mat, seq_len),
          'crnn': load_model(map_item('crnn', paths)),
          'rcnn': load_model(map_item('rcnn', paths))}


def plot_prob(items, probs):
    inds = np.arange(len(items))
    plt.bar(inds, probs, width=0.5)
    plt.xlabel('word')
    plt.ylabel('prob')
    plt.xticks(inds, items)
    plt.show()


def predict(text, name, plot):
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
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
    if name == 'adnn' and plot:
        model = map_item(name + '_plot', models)
        probs = model.predict(pad_seq)[0]
        plot_prob(text, probs[-len(text):])
    return ', '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('adnn: %s' % predict(text, 'adnn', plot=True))
        print('crnn: %s' % predict(text, 'crnn', plot=False))
        print('rcnn: %s' % predict(text, 'rcnn', plot=False))
