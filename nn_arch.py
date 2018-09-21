from keras.layers import Dense, SeparableConv1D, LSTM
from keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization
from keras.layers import Lambda, Concatenate, Masking

import keras.backend as K


def dnn(embed_input, class_num):
    da1 = Dense(200, activation='relu')
    da2 = Dense(200, activation='relu')
    da3 = Dense(class_num, activation='softmax')
    x = Lambda(lambda a: K.mean(a, axis=1))(embed_input)
    x = da1(x)
    x = da2(x)
    x = Dropout(0.5)(x)
    return da3(x)


def cnn(embed_input, class_num):
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    bn = BatchNormalization()
    mp = GlobalMaxPooling1D()
    da1 = Dense(200, activation='relu')
    da2 = Dense(class_num, activation='softmax')
    x1 = ca1(embed_input)
    x1 = bn(x1)
    x1 = mp(x1)
    x2 = ca2(embed_input)
    x2 = bn(x2)
    x2 = mp(x2)
    x3 = ca3(embed_input)
    x3 = bn(x3)
    x3 = mp(x3)
    x = Concatenate()([x1, x2, x3])
    x = da1(x)
    x = Dropout(0.5)(x)
    return da2(x)


def rnn(embed_input, class_num):
    ra = LSTM(200, activation='tanh')
    da = Dense(class_num, activation='softmax')
    x = Masking()(embed_input)
    x = ra(x)
    x = Dropout(0.5)(x)
    return da(x)
