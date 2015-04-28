#! /usr/bin/env python
# pylint: disable=missing-docstring

import numpy
from numpy.testing import assert_array_equal
from nose.tools import assert_equal

import theano

from simplelearn.nodes import (InputNode,
                               RescaleImage,
                               FormatNode,
                               Conv2DLayer,
                               SoftmaxLayer)
from simplelearn.formats import DenseFormat

import pylearn2
from pylearn2.models import mlp
# from pylearn2.models.mlp import MLP, Softmax, ConvRectifiedLinear


def main():
    floatX = theano.config.floatX

    mnist_image_node = InputNode(DenseFormat(axes=('b', '0', '1'),
                                             shape=(-1, 3, 4),
                                             dtype='uint8'))
    scaled_image_node = RescaleImage(mnist_image_node)

    num_filters = 2
    filter_shape = (2, 2)
    pool_shape = (2, 2)
    pool_stride = (2, 2)
    seed = 1234
    conv_uniform_range = .05
    affine_stddev = .05
    num_classes = 2
    batch_size = 2

    def make_sl_model(scaled_image_node):
        '''
        Builds a convlayer-softmaxlayer network on top of mnist_image_node.

        Returns
        -------
        rval: list
          [Conv2DLayer, SoftmaxLayer]. This isn't all the nodes from
          input to output (it omits the scaling and formatting nodes).
        '''

        assert_equal(str(scaled_image_node.output_format.dtype), 'float32')

        rng = numpy.random.RandomState(seed)
        layers = []

        last_node = scaled_image_node
        last_node = RescaleImage(last_node)

        image_dims = mnist_image_node.output_format.shape[1:]

        last_node = FormatNode(last_node,
                               DenseFormat(axes=('b', 'c', '0', '1'),
                                           shape=(-1, 1) + image_dims,
                                           dtype=None),
                               {'b': ('b', 'c')})

        last_node = Conv2DLayer(last_node,
                                filter_shape=filter_shape,
                                num_filters=num_filters,
                                conv_pads='valid',
                                pool_window_shape=pool_shape,
                                pool_strides=pool_stride,
                                pool_pads='pylearn2')
        layers.append(last_node)

        filters = last_node.conv2d_node.filters
        filters.set_value(rng.uniform(low=-conv_uniform_range,
                                      high=conv_uniform_range,
                                      size=filters.get_value().shape))

        last_node = SoftmaxLayer(last_node,
                                 DenseFormat(axes=('b', 'f'),
                                             shape=(-1, num_classes),
                                             dtype=None))
        layers.append(last_node)

        weights = last_node.affine_node.linear_node.params
        weights.set_value(rng.standard_normal(weights.get_value().shape) *
                          affine_stddev)

        return layers

    sl_layers = make_sl_model(mnist_image_node)

    def get_sl_function(sl_layers):
        input_symbol = scaled_image_node.output_symbol
        output_symbols = [layer.output_symbol for layer in sl_layers]
        return theano.function(input_symbol, output_symbols)

    sl_function = get_sl_function(sl_layers)

    def make_pylearn2_model(mnist_image_node):
        layers = []
        layers.append(mlp.ConvRectifiedLinear(layer_name='conv',
                                              tied_b=True,
                                              output_channels=num_filters,
                                              irange=conv_uniform_range,
                                              kernel_shape=filter_shape,
                                              pool_shape=pool_shape,
                                              pool_stride=pool_stride))

        # weights.set_value(sl_weights.get_value())
        # assert_true(numpy.all(biases.get_value() == 0.0))

        layers.append(mlp.Softmax(layer_name="softmax",
                                  n_classes=num_classes,
                                  istdev=affine_stddev))

        # sl_weights = sl_softmax_model[1].affine_node.filters
        # weights.set_value(sl_weights.get_value())
        # assert_true(numpy.all(biases.get_value() == 0.0))

        image_shape = mnist_image_node.output_format.shape[1:]

        result = mlp.MLP(
            layers=layers,
            seed=seed,
            batch_size=batch_size,
            input_space=pylearn2.space.Conv2DSpace(shape=image_shape,
                                                   num_channels=1))

        weights, biases = layers[-1].get_params()
        sl_weights = sl_layers[0].conv2d_node.filters
        assert_array_equal(weights.get_value(), sl_weights.get_value())
        sl_biases = sl_layers[0].bias_node.params
        assert_array_equal(biases.get_value(), sl_biases.get_value())

        biases, weights = layers[-1].get_params()
        sl_weights = sl_layers[1].affine_node.linear_node.params
        assert_array_equal(weights.get_value(), sl_weights.get_value())
        sl_biases = sl_layers[1].affine_node.bias_node.params
        assert_array_equal(biases.get_value(), sl_biases.get_value())

        return result

    pl_mlp = make_pylearn2_model(mnist_image_node)

    def get_mlp_function(pl_mlp):
        input_state = scaled_image_node.output_symbol
        outputs = pl_mlp.fprop(state_below=input_state,
                               return_all=True)
        return theano.function(input_state, outputs)

    pl_function = get_mlp_function(pl_mlp)

    image_batch = mnist_image_node.output_format.make_batch(
        batch_size=batch_size,
        is_symbolic=False)

    sl_outputs = sl_function(image_batch)
    pl_outputs = pl_function(image_batch)

    assert_array_equal(sl_outputs, pl_outputs)


if __name__ == '__main__':
    main()
