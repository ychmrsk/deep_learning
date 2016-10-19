#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from util.layers import *
from util.neural_network import BaseNetwork
from mnist import load_mnist
from util.trainer import Trainer

class SampleNetwork(BaseNetwork):
    
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, activation='relu',
                 weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ratio=0.5,  use_batchnorm=False):
        BaseNetwork.__init__(self, input_dim[1], [hidden_size], output_size,
                             weight_init_std, weight_decay_lambda,
                             use_dropout, dropout_ratio, use_batchnorm)

        input_ch, input_size, _ = input_dim
        filter_num, filter_size = conv_param['filter_num'], conv_param['filter_size']
        pad, stride = conv_param['pad'], conv_param['stride']
        conv_output_size = (input_size - filter_size + 2 * pad) / stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # convolution - activation - pooling
        BaseNetwork._generate_convolution_layer(self, 1, input_ch, input_size, conv_param, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 1, activation)
        BaseNetwork._generate_pooling_layer(self, 1, pool_h=2, pool_w=2, stride=2)
        # hidden layer
        BaseNetwork._generate_affine_layer(self, 2, pool_output_size, hidden_size, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 2, activation)
        # output layer
        BaseNetwork._generate_affine_layer(self, 3, hidden_size, output_size, weight_init_std)


if __name__ == '__main__':
    # load MNIST dataset
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    # data reduction if need
    x_train, t_train = x_train[:5000], t_train[:5000]
    x_test, t_test = x_test[:1000], t_test[:1000]
    
    # network setting
    network = SampleNetwork(input_dim=(1, 28, 28),
                            conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                            hidden_size=100, output_size=10, weight_init_std=0.01)
    
    # training
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=20, mini_batch_size=100,
                      optimizer='SGD', optimizer_param={'lr': 0.1},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()
    train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

    # save parameters
    network.save_params('DIRNAME_FILENAME')
    print('saved network parameters.')

    # draw graph
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
    
