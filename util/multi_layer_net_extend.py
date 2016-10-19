#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from util.layers import *
from util.gradient import numerical_gradient


class MultiLayerNetExtend(object):
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ratio=0.5, use_batchnorm=False):
        self.size_list = [input_size] + hidden_size_list + [output_size]
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.layers = OrderedDict()
        self.layers_info = []

        # hidden layer
        for idx in range(1, len(self.size_list) - 1):
            self.__generate_affine_layer(
                idx, self.size_list[idx-1], self.size_list[idx], weight_init_std)
            if self.use_batchnorm:
                self.__generate_batchnorm_layer(idx, self.size_list[idx])
            self.__generate_activation_layer(idx, activation)
            if self.use_dropout:
                self.__generate_dropout_layer(idx, dropout_ratio)

        # output layer
        idx = len(self.size_list) - 1
        self.__generate_affine_layer(
            idx, self.size_list[idx-1], self.size_list[idx], weight_init_std)
        self.last_layer = SoftmaxWithLoss()

    def __generate_activation_layer(self, idx, activation):
        activation_layer = {'sigmoid':Sigmoid, 'relu':Relu}
        self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
        self.layers_info.append(('Activation_function', idx))
        
    def __generate_dropout_layer(self, idx, dropout_ratio):
        self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio)
        self.layers_info.append(('Dropout', idx))
        
    def __generate_batchnorm_layer(self, idx, node_num):
        self.params['gamma' + str(idx)] = np.ones(node_num)
        self.params['beta' + str(idx)] = np.zeros(node_num)
        self.layers['BatchNorm' + str(idx)] = BatchNormalization(
            self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
        self.layers_info.append(('BatchNorm', idx))

    def __generate_affine_layer(self, idx, prev_node_num, curr_node_num, weight_init_std):
        # setting weight scailing
        scale = weight_init_std
        if str(weight_init_std).lower() in ('relu', 'he'):
            scale = np.sqrt(2.0 / prev_node_num)
        elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
            scale = np.sqrt(1.0 / prev_node_num)
        # initialize weights
        self.params['W' + str(idx)] = scale * np.random.randn(prev_node_num, curr_node_num)
        self.params['b' + str(idx)] = np.zeros(curr_node_num)
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        self.layers_info.append(('Affine', idx))

    def predict(self, x, train_flg=False):
        for layer, idx in self.layers_info:
            if layer == 'Dropout' or layer == 'BatchNorm':
                x = self.layers[layer + str(idx)].forward(x, train_flg)
            else:
                x = self.layers[layer + str(idx)].forward(x)
        return x
                
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        weight_decay = 0

        for layer, idx in self.layers_info:
            if layer == 'Affine':
                W = self.layers[layer + str(idx)].W
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        return self.last_layer.forward(y, t) + weight_decay
        
#    def accuracy(self, x, t):
#        y = self.predict(x, train_flg=True)
#        y = np.argmax(y, axis=1)
#        if t.ndim != 1:
#            t = np.argmax(t, axis=1)
#        accuracy = np.sum(y == t) / float(x.shape[0])
#        return accuracy

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size: (i+1) * batch_size]
            tt = t[i * batch_size: (i+1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for layer, idx in self.layers_info:
            if layer == 'Affine':
                grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
                grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            elif layer == 'BatchNorm':
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])
        return grads

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for layer, idx in self.layers_info:
            if layer == 'Affine':
                grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + \
                                        self.weight_decay_lambda * self.params['W' + str(idx)]
                grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
            elif layer == 'BatchNorm':
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
        return grads

