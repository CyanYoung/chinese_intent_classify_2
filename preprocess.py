import os

import re

from random import shuffle, randint

from util import load_word_re, load_type_re, load_pair, word_replace


def drop(words, bound):
    ind = randint(0, bound)
    words.pop(ind)
    return ''.join(words)


def swap(words, bound):
    ind1, ind2 = randint(0, bound), randint(0, bound)
    words[ind1], words[ind2] = words[ind2], words[ind1]
    return ''.join(words)


def copy(words, bound):
    ind1, ind2 = randint(0, bound), randint(0, bound)
    words.insert(ind1, words[ind2])
    return ''.join(words)


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

aug_rate = 2

funcs = [drop, swap, copy]


def sync_shuffle(texts, labels):
    texts_labels = list(zip(texts, labels))
    shuffle(texts_labels)
    return zip(*texts_labels)


def augment(texts, labels):
    aug_texts, aug_labels = list(), list()
    for text, label in zip(texts, labels):
        bound = len(text) - 1
        if bound > 0:
            for func in funcs:
                for _ in range(aug_rate):
                    words = list(text)
                    aug_texts.append(func(words, bound))
                    aug_labels.append(label)
    return sync_shuffle(aug_texts, aug_labels)


def save(path, texts, labels):
    aug_texts, aug_labels = augment(texts, labels)
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(aug_texts, aug_labels):
            f.write(text + ',' + label + '\n')


def clean(text):
    text = re.sub(stop_word_re, '', text)
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    return word_replace(text, syno_dict)


def prepare(path_univ_dir, path_train, path_test):
    text_set = set()
    texts, labels = list(), list()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = line.strip().lower()
                text = clean(text)
                if text and text not in text_set:
                    text_set.add(text)
                    texts.append(text)
                    labels.append(label)
    texts, labels = sync_shuffle(texts, labels)
    bound = int(len(texts) * 0.9)
    save(path_train, texts[:bound], labels[:bound])
    save(path_test, texts[bound:], labels[bound:])


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    prepare(path_univ_dir, path_train, path_test)
