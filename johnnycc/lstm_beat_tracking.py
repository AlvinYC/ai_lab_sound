from __future__ import print_function
import librosa
import numpy as np
import os
import sys
import argparse

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.contrib.keras.python.keras.layers.recurrent import LSTM
from tensorflow.contrib.keras.python.keras.layers.wrappers import TimeDistributed
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras import backend as K

from mirex2016_dataset import load_trainXY
from keras import backend as K

def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision


def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall

parser = argparse.ArgumentParser()
#x431x128_y40x431
parser.add_argument('--datadir', type=str, default='./dataset/mirex_beat_tracking_2016/train/x862x1025_y40x862/', help='train data path')
parser.add_argument('--logdir', type=str, default='./dataset/log', help='Tensorboard log path')
parser.add_argument('--epochs', type=int, default=200, help='the number of times to iterate over the training data')
parser.add_argument('-lr', type=lambda lr_str: [int(lr) for lr in lr_str.split(',')],
    default=[10, 1, 0.1, 0.01, 0.006, 0.003, 0.001, 0.0006, 0.0003, 0.0001, 0.00005, 0.00001], help='learning rate')
args = parser.parse_args()
trainXY_path = args.datadir
tb_logdir = args.logdir
epochs = args.epochs
learning_rates = args.lr
loss_function = 'binary_crossentropy'

trainX, trainY, n_of_frames, n_of_freq_bins = load_trainXY(trainXY_path)
trainX.resize(n_of_frames, 1, n_of_freq_bins)
trainY.resize(n_of_frames, 1, 1)

for lr in learning_rates:
    model = Sequential()
    # build a LSTM RNN
    model.add(LSTM(
        units=128,  # number of neuons? original is 100, why 128? because the number of mel freq bins is 128
        input_shape=(1, n_of_freq_bins),
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        use_bias=True,
        return_sequences=True,      # True: output at all steps. False: output as last step.
        stateful=False,              # True: the final state of batch1 is feed into the initial state of batch2
    ))
    model.add(Dense(1))
    rmsprop = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss=loss_function, optimizer=rmsprop, metrics=['accuracy', recall, precision, f1_score])

    tb_logpath = tb_logdir + '/%s_rmsprop_lr_%f' % (loss_function, lr)
    tb = tf.contrib.keras.callbacks.TensorBoard(log_dir=tb_logpath, histogram_freq=0, write_graph=True, write_images=True)
    model.fit(trainX, trainY,
        batch_size=n_of_frames,
        epochs=epochs,
        validation_split=0.2,
        validation_data=None,
        shuffle=True,
        callbacks=[tb])

#pred = model.predict(X_batch, BATCH_SIZE)
