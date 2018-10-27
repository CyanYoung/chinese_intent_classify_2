from keras.layers import Dense, SeparableConv1D, LSTM, Activation
from keras.layers import Dropout, TimeDistributed, GlobalMaxPooling1D, Bidirectional
from keras.layers import Lambda, Flatten, RepeatVector, Permute, Concatenate, Multiply

import keras.backend as K


embed_len = 200


def attend(x, embed_len):
    da = Dense(200, activation='tanh', name='attend1')
    dn = Dense(1, activation=None, name='attend2')
    tn = TimeDistributed(dn)
    softmax = Activation('softmax')
    mean = Lambda(lambda a: K.mean(a, axis=1))
    p = da(x)
    p = tn(p)
    p = Flatten()(p)
    p = softmax(p)
    p = RepeatVector(embed_len)(p)
    p = Permute((2, 1))(p)
    x = Multiply()([x, p])
    return mean(x)


def adnn(embed_input, class_num):
    da1 = Dense(200, activation='relu', name='encode1')
    da2 = Dense(200, activation='relu', name='encode2')
    da3 = Dense(class_num, activation='softmax', name='classify')
    x = attend(embed_input, embed_len)
    x = da1(x)
    x = da2(x)
    x = Dropout(0.5)(x)
    return da3(x)


def adnn_plot(x):
    da = Dense(200, activation='tanh', name='attend1')
    dn = Dense(1, activation=None, name='attend2')
    tn = TimeDistributed(dn)
    softmax = Activation('softmax')
    p = da(x)
    p = tn(p)
    p = Flatten()(p)
    return softmax(p)


def crnn(embed_input, class_num):
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    ra = LSTM(200, activation='tanh')
    da = Dense(class_num, activation='softmax')
    x1 = ca1(embed_input)
    x2 = ca2(embed_input)
    x3 = ca3(embed_input)
    x = Concatenate()([x1, x2, x3])
    x = ra(x)
    x = Dropout(0.5)(x)
    return da(x)


def rcnn(embed_input, class_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    mp = GlobalMaxPooling1D()
    da1 = Dense(200, activation='relu')
    da2 = Dense(class_num, activation='softmax')
    x = ba(embed_input)
    x1 = ca1(x)
    x1 = mp(x1)
    x2 = ca2(x)
    x2 = mp(x2)
    x3 = ca3(x)
    x3 = mp(x3)
    x = Concatenate()([x1, x2, x3])
    x = da1(x)
    x = Dropout(0.5)(x)
    return da2(x)
