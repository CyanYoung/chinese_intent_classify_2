import pickle as pk

import re

import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from util import load_word_re, load_type_re, load_word_pair, word_replace
from util import map_path, map_model


seq_len = 30

path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
path_word2ind = 'model/word2ind.pkl'
path_label_ind = 'feat/label_ind.pkl'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_word_pair(path_homo)
syno_dict = load_word_pair(path_syno)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = dict()
for label, ind in label_inds.items():
    ind_labels[ind] = label

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}

models = {'dnn': load_model(map_path('dnn', paths)),
          'cnn': load_model(map_path('cnn', paths)),
          'rnn': load_model(map_path('rnn', paths))}


def predict(text, name):
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_model(name, models)
    probs = model.predict(pad_seq)[0]
    max_ind = np.argmax(probs)
    return ind_labels[max_ind]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('dnn: %s' % predict(text, 'dnn'))
        print('cnn: %s' % predict(text, 'cnn'))
        print('rnn: %s' % predict(text, 'rnn'))
