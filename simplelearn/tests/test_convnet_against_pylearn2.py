'''
Runs Simplelearn and Pylearn2 versions of the MNIST demo convnet in
parallel.
'''

# pylint: disable=missing-docstring

from __future__ import print_function

import os
import numpy
from nose.tools import assert_equal, assert_is_instance
from numpy.testing import assert_array_equal, assert_allclose
import theano
from simplelearn.formats import DenseFormat
from simplelearn.nodes import (Conv2DLayer,
                               SoftmaxLayer,
                               RescaleImage,
                               FormatNode,
                               CrossEntropy)
from simplelearn.data.mnist import load_mnist
from simplelearn.data.dataset import Dataset
from simplelearn.utils import safe_izip


import pylearn2
from pylearn2.config import yaml_parse

import pdb

def split_dataset(dataset, first_size):
    first_tensors = [t[:first_size, ...] for t in dataset.tensors]
    first_dataset = Dataset(tensors=first_tensors,
                            names=dataset.names,
                            formats=dataset.formats)

    second_tensors = [t[first_size:, ...] for t in dataset.tensors]
    second_dataset = Dataset(tensors=second_tensors,
                             names=dataset.names,
                             formats=dataset.formats)

    return first_dataset, second_dataset


def make_sl_model(mnist_image_node, rng):
    layers = []  # to return

    # CNN specs
    filter_counts = [64, 64]
    filter_shapes = [(5, 5), (5, 5)]
    pool_shapes = [(4, 4), (4, 4)]
    pool_strides = [(2, 2), (2, 2)]
    conv_half_range = .05
    softmax_stddev = .05
    # end CNN specs

    num_rows, num_columns = mnist_image_node.output_format.shape[1:]

    last_node = mnist_image_node
    last_node = RescaleImage(last_node)
    last_node = FormatNode(last_node,
                           # use b01c for pl model
                           DenseFormat(axes=('b', 'c', '0', '1'),
                                       shape=(-1, 1, num_rows, num_columns),
                                       dtype=None),
                           axis_map={'b': ('b', 'c')})

    for (filter_count,
         filter_shape,
         pool_shape,
         pool_stride) in safe_izip(filter_counts,
                                   filter_shapes,
                                   pool_shapes,
                                   pool_strides):
        last_node = Conv2DLayer(last_node,
                                filter_shape,
                                filter_count,
                                conv_pads='valid',
                                pool_window_shape=pool_shape,
                                pool_strides=pool_stride,
                                pool_pads='pylearn2',
                                conv_mode='conv')

        weights = last_node.conv2d_node.filters
        weight_values = weights.get_value()
        weight_values[...] = rng.uniform(low=-conv_half_range,
                                         high=conv_half_range,
                                         size=weight_values.shape)
        weights.set_value(weight_values)

        # Don't init biases.

        layers.append(last_node)

    num_classes = 10
    last_node = SoftmaxLayer(last_node,
                             DenseFormat(axes=('b', 'f'),
                                         shape=(-1, num_classes),
                                         dtype=None),
                             input_to_bf_map={('0', '1', 'c'): 'f'})

    weights = last_node.affine_node.linear_node.params
    weight_values = weights.get_value()
    weight_values[...] = (rng.standard_normal(weight_values.shape)
                          * softmax_stddev)
    weights.set_value(weight_values)

    layers.append(last_node)

    return layers

def make_pl_model():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'mnist_conv_pylearn2_model.yaml')
    with open(file_path) as yaml_file:
        mlp = yaml_parse.load(yaml_file)

    return mlp

def get_sl_grads_function(image_node, label_node, sl_layers):
    '''
    Compiles a theano function that returns all gradients.

    Returns a function that takes MNIST images and labels,
    and outputs the loss w.r.t. the simplelearn layers' params.

    The function returns a tuple of 6 elements: [dW_0, db_0, dW_1, db_1, dW_2,
    db_2], where dW_i is the gradient wrt the i'th layer's weights or filters,
    and db_i is the gradient wrt the i'th layer's biases.
    '''
    assert_equal(len(sl_layers), 3)

    cost_node = CrossEntropy(sl_layers[-1], label_node)
    scalar_cost = cost_node.output_symbol.mean()

    params = []
    for conv_layer in sl_layers[:2]:
        params.append(conv_layer.conv2d_node.filters)
        params.append(conv_layer.bias_node.params)

    params.append(sl_layers[2].affine_node.linear_node.params)
    params.append(sl_layers[2].affine_node.bias_node.params)
    assert_equal(len(params), 6)

    for pi, param in enumerate(params):
        layer_index = pi // 2
        is_bias = bool(pi % 2)

        if layer_index < 2:
            assert_equal(param.ndim, 1 if is_bias else 4)
        else:
            assert_equal(param.ndim, 1 if is_bias else 2)

    grads = theano.gradient.grad(scalar_cost, params)
    return theano.function([image_node.output_symbol,
                            label_node.output_symbol],
                           grads)

def get_onehot_labels_symbol(indices_symbol):
    '''
    Converts a vector of label indices to a matrix of one-hot rows.

    (This operates on Theano variables, not numpy arrays.)
    '''
    T = theano.tensor
    batch_size = indices_symbol.shape[0]
    result = T.zeros((batch_size, 10), dtype=indices_symbol.dtype)

    one_hot = T.set_subtensor(result[T.arange(batch_size), indices_symbol],
                              1)

    return one_hot

def get_pl_grads_function(mnist_image_node,
                          label_node,
                          mlp_output_symbol,
                          mlp):
    '''
    Does the same as get_sl_grads_function, but for the Pylearn2 model.

    See docstring for get_sl_grads_function.
    '''
    assert_is_instance(mlp, pylearn2.models.mlp.MLP)
    assert_equal(len(mlp.layers), 3)

    params = []
    for conv_layer in mlp.layers[:2]:
        params.extend(conv_layer.transformer.get_params())
        params.append(conv_layer.b)

    bias, weights = mlp.layers[2].get_params()
    params.append(weights)
    params.append(bias)

    for pi, param in enumerate(params):
        layer_index = pi // 2
        is_bias = bool(pi % 2)

        if layer_index < 2:
            assert_equal(param.ndim, 1 if is_bias else 4)
        else:
            assert_equal(param.ndim, 1 if is_bias else 2)

    onehot_labels_symbol = get_onehot_labels_symbol(label_node.output_symbol)

    scalar_cost = mlp.cost(onehot_labels_symbol,
                           mlp_output_symbol)

    grads = theano.gradient.grad(scalar_cost, params)
    return theano.function([mnist_image_node.output_symbol,
                            label_node.output_symbol],
                           grads)


def test_convnet_against_pylearn2():

    training_set = load_mnist()[0]
    training_set, validating_set = split_dataset(training_set, 50000)
    assert_equal(validating_set.tensors[0].shape[0], 10000)

    batch_size = 100

    training_iterator = training_set.iterator('sequential', batch_size)
    mnist_image_node, mnist_label_node = training_iterator.make_input_nodes()

    seed = 1234  # the same one used in ./mnist_conv_pylearn2_model.yaml

    sl_layers = make_sl_model(mnist_image_node, numpy.random.RandomState(seed))

    image_batch, label_batch = training_iterator.next()

    sl_layers_function = theano.function([mnist_image_node.output_symbol],
                                         [x.output_symbol for x in sl_layers])
    sl_layer_outputs = sl_layers_function(image_batch)

    pl_model = make_pl_model()
    float_image_node = RescaleImage(mnist_image_node)
    num_rows = mnist_image_node.output_format.shape[1]
    num_cols = mnist_image_node.output_format.shape[2]
    pl_image_node = FormatNode(float_image_node,
                               DenseFormat(axes=('b', '0', '1', 'c'),
                                           shape=(-1, num_rows, num_cols, 1),
                                           dtype=None),
                               axis_map={'1': ('1', 'c')})

    pl_layer_symbols = pl_model.fprop(state_below=pl_image_node.output_symbol,
                                      return_all=True)
    pl_layers_function = theano.function([mnist_image_node.output_symbol],
                                         pl_layer_symbols)

    pl_layer_outputs = pl_layers_function(image_batch)

    for pl_layer_output, sl_layer_output in safe_izip(pl_layer_outputs,
                                                      sl_layer_outputs):
        # On some graphics cards (e.g. NVidia GTX 780), sl and pl outputs are
        # equal.  On others (e.g. NVidia GT 650M), they're close but not equal.
        assert_allclose(pl_layer_output, sl_layer_output, atol=1e-5)
        # assert_array_equal(pl_layer_output, sl_layer_output)

    sl_grads_function = get_sl_grads_function(mnist_image_node,
                                              mnist_label_node,
                                              sl_layers)
    pl_grads_function = get_pl_grads_function(mnist_image_node,
                                              mnist_label_node,
                                              pl_layer_symbols[-1],
                                              pl_model)

    sl_grads = sl_grads_function(image_batch, label_batch)
    pl_grads = pl_grads_function(image_batch, label_batch)

    for sl_grad, pl_grad in safe_izip(sl_grads, pl_grads):

        # Can't use assert_array_equal here. They won't be equal, since they
        # use different implementations of cross-entropy
        assert_allclose(sl_grad, pl_grad, atol=1e-5)
