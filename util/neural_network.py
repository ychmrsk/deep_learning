#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from collections import OrderedDict
from util.layers import *
from util.gradient import numerical_gradient


class BaseNetwork(object):

    def __init__(self, input_size, hidden_size_list, output_size,
                 weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ratio=0.5, use_batchnorm=False):
        
        self.size_list = [input_size] + hidden_size_list + [output_size]
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.use_batchnorm = use_batchnorm

        self.params = {}
        self.layers = OrderedDict()
        self.layers_info = []

        self.last_layer = SoftmaxWithLoss()


    def _scale_optimizer(self, weight_init_std, prev_node_num):
        scale = weight_init_std
        if str(weight_init_std).lower() in ('relu', 'he'):
            scale = np.sqrt(2.0 / prev_node_num)
        elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
            scale = np.sqrt(1.0 / prev_node_num)
        return scale
        
    def _generate_affine_layer(self, idx, prev_node_num, curr_node_num, weight_init_std):
        scale = self._scale_optimizer(weight_init_std, prev_node_num)
        self.params['W' + str(idx)] = scale * np.random.randn(prev_node_num, curr_node_num)
        self.params['b' + str(idx)] = np.zeros(curr_node_num)
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        self.layers_info.append(('Affine', idx))

    def _generate_batchnorm_layer(self, idx, node_num):
        self.params['gamma' + str(idx)] = np.ones(node_num)
        self.params['beta' + str(idx)] = np.zeros(node_num)
        self.layers['BatchNorm' + str(idx)] = BatchNormalization(
            self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
        self.layers_info.append(('BatchNorm', idx))

    def _generate_activation_layer(self, idx, activation):
        activation_layer = {'sigmoid':Sigmoid, 'relu':Relu}
        self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
        self.layers_info.append(('Activation_function', idx))
        
    def _generate_dropout_layer(self, idx, dropout_ratio):
        self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio)
        self.layers_info.append(('Dropout', idx))
        
    def _generate_convolution_layer(self, idx, pre_ch_num, pre_size, param, weight_init_std):
        filter_num, filter_size, pad, stride = \
            param['filter_num'], param['filter_size'], param['pad'], param['stride']
        pre_node_num = (1 + (pre_size - filter_size + 2*pad) / stride) ** 2
        scale = self._scale_optimizer(weight_init_std, pre_node_num)
        self.params['W' + str(idx)] = scale * np.random.randn(filter_num, pre_ch_num,
                                                              filter_size, filter_size)
        self.params['b' + str(idx)] = scale * np.zeros(filter_num)
        self.layers['Conv' + str(idx)] = Convolution(
            self.params['W' + str(idx)], self.params['b' + str(idx)], stride, pad)
        self.layers_info.append(('Conv', idx))

    def _generate_pooling_layer(self, idx, pool_h, pool_w, stride):
        self.layers['Pool' + str(idx)] = Pooling(pool_h, pool_w, stride)
        self.layers_info.append(('Pool', idx))

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
            if layer == 'Affine' or layer == 'Conv':
                W = self.layers[layer + str(idx)].W
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        
        return self.last_layer.forward(y, t) + weight_decay

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
            if layer == 'Affine' or layer == 'Conv':
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
            if layer == 'Affine' or layer == 'Conv':
                grads['W' + str(idx)] = self.layers[layer + str(idx)].dW + \
                                        self.weight_decay_lambda * self.params['W' + str(idx)]
                grads['b' + str(idx)] = self.layers[layer + str(idx)].db
            elif layer == 'BatchNorm':
                grads['gamma' + str(idx)] = self.layers[layer + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers[layer + str(idx)].dbeta
        return grads

    def save_params(self, file_name='params'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(params, f)
        info = self.layers_info
        with open(file_name + '_info.pkl', 'wb') as f:
            pickle.dump(info, f)
            
    def load_params(self, file_name='params'):
        with open(file_name + '.pkl', 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        with open(file_name + '_info.pkl', 'rb') as f:
            self.layers_info = pickle.load(f)
        for layer, idx in self.layers_info:
            if layer == 'Affine' or layer == 'Conv':
                self.layers[layer + str(idx)].W = self.params['W' + str(idx)]
                self.layers[layer + str(idx)].b = self.params['b' + str(idx)]
            if layer == 'BatchNorm':
                self.layers[layer + str(idx)].gamma = self.params['gamma' + str(idx)]
                self.layers[layer + str(idx)].beta = self.params['beta' + str(idx)]


class SimpleConvNet(BaseNetwork):

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, activation='relu', weight_init_std='relu',
                 weight_decay_lambda=0, use_dropout=False, dropout_ratio=0.5, use_batchnorm=False):
        BaseNetwork.__init__(self, input_dim[1], [hidden_size], output_size,
                             weight_init_std, weight_decay_lambda,
                             use_dropout, dropout_ratio, use_batchnorm)
        input_ch, input_size, _ = input_dim
        filter_num, filter_size = conv_param['filter_num'], conv_param['filter_size']
        pad, stride = conv_param['pad'], conv_param['stride']
        conv_output_size = (input_size - filter_size + 2*pad) / stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # convolution - activation - pooling
        BaseNetwork._generate_convolution_layer(self, 1, input_ch, input_size,
                                                 conv_param, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 1, activation)
        BaseNetwork._generate_pooling_layer(self, 1, pool_h=2, pool_w=2, stride=2)
        # hidden layer
        BaseNetwork._generate_affine_layer(self, 2, pool_output_size,
                                            hidden_size, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 2, activation)
        # output layer
        BaseNetwork._generate_affine_layer(self, 3, hidden_size, output_size, weight_init_std)


class MultiLayerNetExtend(BaseNetwork):

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ratio=0.5, use_batchnorm=False):
        BaseNetwork.__init__(self, input_size, hidden_size_list, output_size,
                             weight_init_std, weight_decay_lambda,
                             use_dropout, dropout_ratio, use_batchnorm)
        # hidden layers
        for idx in range(1, len(self.size_list) - 1):
            BaseNetwork._generate_affine_layer(
                self, idx, self.size_list[idx-1], self.size_list[idx], weight_init_std)
            if self.use_batchnorm:
                BaseNetwork._generate_batchnorm_layer(self, idx, self.size_list[idx])
            BaseNetwork._generate_activation_layer(self, idx, activation)
            if self.use_dropout:
                BaseNetwork._generate_dropout_layer(self, idx, dropout_ratio)
        # output layer
        idx = len(self.size_list) - 1
        BaseNetwork._generate_affine_layer(
            self, idx, self.size_list[idx-1], self.size_list[idx], weight_init_std)

class MultiLayerNet(MultiLayerNetExtend):

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        MultiLayerNetExtend.__init__(self, input_size, hidden_size_list, output_size,
                                     activation, weight_init_std, weight_decay_lambda,
                                     use_dropout=False, dropout_ratio=0.5, use_batchnorm=False)

        
class DeepConvNet(BaseNetwork):

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10,
                 activation='relu', weight_init_std='relu'):

        BaseNetwork.__init__(self, input_dim[1], [hidden_size], output_size,
                             weight_init_std='relu', weight_decay_lambda=0,
                             use_dropout=True, dropout_ratio=0.5, use_batchnorm=False)

        ch_and_size = [(None, None)]  # for index.0
        in_ch, in_size, _ = input_dim
        ch_and_size.append((in_ch, in_size))  # for index.1
        for idx, cp in enumerate([conv_param_1, conv_param_2, conv_param_3,
                                  conv_param_4, conv_param_5, conv_param_6], 1):
            out_ch = cp['filter_num']
            out_size = 1 + (in_size + 2 * cp['pad'] - cp['filter_size']) / cp['stride']
            if idx == 2 or idx == 4 or idx == 6:  # pooling
                out_size /= 2
            ch_and_size.append((out_ch, out_size))
            in_ch, in_size = out_ch, out_size

        # layer 1
        BaseNetwork._generate_convolution_layer(
            self, 1, ch_and_size[1][0], ch_and_size[1][1], conv_param_1, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 1, activation)
        # layer 2
        BaseNetwork._generate_convolution_layer(
            self, 2, ch_and_size[2][0], ch_and_size[2][1], conv_param_2, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 2, activation)
        BaseNetwork._generate_pooling_layer(self, 2, pool_h=2, pool_w=2, stride=2)
        # layer 3
        BaseNetwork._generate_convolution_layer(
            self, 3, ch_and_size[3][0], ch_and_size[3][1], conv_param_3, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 3, activation)
        # layer 4
        BaseNetwork._generate_convolution_layer(
            self, 4, ch_and_size[4][0], ch_and_size[4][1], conv_param_4, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 4, activation)
        BaseNetwork._generate_pooling_layer(self, 4, pool_h=2, pool_w=2, stride=2)
        # layer 5
        BaseNetwork._generate_convolution_layer(
            self, 5, ch_and_size[5][0], ch_and_size[5][1], conv_param_5, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 5, activation)
        # layer 6
        BaseNetwork._generate_convolution_layer(
            self, 6, ch_and_size[6][0], ch_and_size[6][1], conv_param_6, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 6, activation)
        BaseNetwork._generate_pooling_layer(self, 6, pool_h=2, pool_w=2, stride=2)
        # layer 7
        conv_out_size = ch_and_size[7][0] * ch_and_size[7][1] * ch_and_size[7][1]
        BaseNetwork._generate_affine_layer(self, 7, conv_out_size, hidden_size, weight_init_std)
        BaseNetwork._generate_activation_layer(self, 7, activation)
        BaseNetwork._generate_dropout_layer(self, 7, dropout_ratio=0.5)
        # layer 8
        BaseNetwork._generate_affine_layer(self, 8, hidden_size, output_size, weight_init_std)
        BaseNetwork._generate_dropout_layer(self, 8, dropout_ratio=0.5)
        
