#! /usr/bin/env python

'''
Script for running Simplelearn and Pylearn2 versions of the MNIST demo
convnet in parallel.
'''

# pylint: disable=missing-docstring

import numpy
from nose.tools import assert_equal
import theano
from simplelearn.formats import DenseFormat
from simplelearn.nodes import (Conv2DLayer,
                               SoftmaxLayer,
                               RescaleImage,
                               FormatNode)
from simplelearn.data.mnist import load_mnist
from simplelearn.data.dataset import Dataset
from simplelearn.utils import safe_izip

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
                                pool_pads='pylearn2')

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


def main():

    training_set = load_mnist()[0]
    training_set, validating_set = split_dataset(training_set, 50000)
    assert_equal(validating_set.tensors[0].shape[0], 10000)

    batch_size = 100

    training_iterator = training_set.iterator('sequential', batch_size)
    mnist_image_node = training_iterator.make_input_nodes()[0]

    seed = 1234

    sl_layers = make_sl_model(mnist_image_node,
                                    numpy.random.RandomState(seed))

    pdb.set_trace()

    image_batch = training_iterator.next()[0]

    layer_1_conv_function = theano.function([mnist_image_node.output_symbol],
                                            sl_layers[1].conv2d_node.output_symbol)

    layer_1_conv_output = layer_1_conv_function(image_batch)

    layer_1_bias_function = theano.function([mnist_image_node.output_symbol],
                                            sl_layers[1].bias_node.output_symbol)

    pdb.set_trace()
    # crashes here
    layer_1_bias_output = layer_1_bias_function(image_batch)

    layer_1_function = theano.function([mnist_image_node.output_symbol],
                                       sl_layers[1].output_symbol)

    layer_1_function(image_batch)


    # layer_2_linear_function = theano.function([mnist_image_node.output_symbol],
    #                                           sl_layers[2].affine_node.linear_node.output_symbol)

    # layer_2_linear_function(image_batch)

    # sl_model_function = theano.function([mnist_image_node.output_symbol],
    #                                     sl_layers[-1].output_symbol)
    # # This should crash... it does!
    # sl_model_function(image_batch)

if __name__ == '__main__':
    main()
