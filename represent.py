import pickle as pk

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from util import flat_read


embed_len = 200
max_vocab = 5000
seq_len = 30

path_word_vec = 'feat/word_vec.pkl'
path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'


def embed(sents, path_word2ind, path_word_vec, path_embed):
    model = Tokenizer(num_words=max_vocab, filters='', char_level=True)
    model.fit_on_texts(sents)
    word_inds = model.word_index
    with open(path_word2ind, 'wb') as f:
        pk.dump(model, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab + 1, len(word_inds) + 1)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def label2ind(labels, path_label_ind):
    labels = sorted(list(set(labels)))
    label_inds = dict()
    for i in range(len(labels)):
        label_inds[labels[i]] = i
    with open(path_label_ind, 'wb') as f:
        pk.dump(label_inds, f)


def align(sents, labels, path_sent, path_label, mode):
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    seqs = model.texts_to_sequences(sents)
    pad_seqs = pad_sequences(seqs, maxlen=seq_len)
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    class_num = len(label_inds)
    inds = list()
    for label in labels:
        inds.append(label_inds[label])
    if mode == 'train':
        inds = to_categorical(inds, num_classes=class_num)
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)
    with open(path_label, 'wb') as f:
        pk.dump(inds, f)


def vectorize(path_data, path_sent, path_label, mode):
    sents = flat_read(path_data, 'text')
    labels = flat_read(path_data, 'label')
    if mode == 'train':
        embed(sents, path_word2ind, path_word_vec, path_embed)
        label2ind(labels, path_label_ind)
    align(sents, labels, path_sent, path_label, mode)


if __name__ == '__main__':
    path_data = 'data/train.csv'
    path_sent = 'feat/sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    vectorize(path_data, path_sent, path_label, 'train')
    path_data = 'data/test.csv'
    path_sent = 'feat/sent_test.pkl'
    path_label = 'feat/label_test.pkl'
    vectorize(path_data, path_sent, path_label, 'test')
