import pandas as pd
import pickle as pk

import numpy as np

from keras.models import load_model

from sklearn.metrics import accuracy_score

from util import flat_read, map_path, map_model


path_test = 'data/test.csv'
path_sent = 'feat/sent_test.pkl'
path_label = 'feat/label_test.pkl'
path_label_ind = 'feat/label_ind.pkl'
texts = flat_read(path_test, 'text')
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)
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


def test(sents, labels, name):
    model = map_model(name, models)
    probs = model.predict(sents)
    preds = np.argmax(probs, axis=1)
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(labels, preds)))
    for text, label, pred in zip(texts, labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, ind_labels[label], ind_labels[pred]))


if __name__ == '__main__':
    test(sents, labels, 'dnn')
    test(sents, labels, 'cnn')
    test(sents, labels, 'rnn')
