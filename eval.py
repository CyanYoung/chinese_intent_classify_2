import pickle as pk

import numpy as np

from sklearn.metrics import accuracy_score

from classify import ind_labels, models

from util import flat_read, map_item


path_test = 'data/test.csv'
path_sent = 'feat/sent_test.pkl'
path_label = 'feat/label_test.pkl'
texts = flat_read(path_test, 'text')
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, sents, labels):
    model = map_item(name, models)
    probs = model.predict(sents)
    preds = np.argmax(probs, axis=1)
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(labels, preds)))
    for text, label, pred in zip(texts, labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, ind_labels[label], ind_labels[pred]))


if __name__ == '__main__':
    test('adnn', sents, labels)
    test('crnn', sents, labels)
    test('rcnn', sents, labels)
