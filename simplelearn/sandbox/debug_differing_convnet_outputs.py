#! /usr/bin/env python
# pylint: disable=missing-docstring

from __future__ import print_function

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
from simplelearn.utils import safe_izip

import pylearn2
from pylearn2.models import mlp
# from pylearn2.models.mlp import MLP, Softmax, ConvRectifiedLinear

import pdb


def main():
    floatX = theano.config.floatX

    # num_rows = 3
    # num_cols = 4

    num_rows = 28
    num_cols = 28

    mnist_image_node = InputNode(DenseFormat(axes=('b', '0', '1'),
                                             shape=(-1, num_rows, num_cols),
                                             dtype='uint8'))
    scaled_image_node = RescaleImage(mnist_image_node)
    pl_input_node = FormatNode(scaled_image_node,
                               DenseFormat(axes=('b', '0', '1', 'c'),
                                           shape=(-1, num_rows, num_cols, 1),
                                           dtype=None),
                               axis_map={'1': ('1', 'c')})

    sl_input_node = FormatNode(scaled_image_node,
                               DenseFormat(axes=('b', 'c', '0', '1'),
                                           shape=(-1, 1, num_rows, num_cols),
                                           dtype=None),
                               axis_map={'b': ('b', 'c')})

    num_filters = 2
    filter_shape = (2, 2)
    pool_shape = (2, 2)
    pool_stride = (2, 2)
    seed = 1234
    conv_uniform_range = .05
    affine_stddev = .05
    num_classes = 2
    batch_size = 2

    def make_sl_model(sl_input_node):
        """
        Builds a convlayer-softmaxlayer network on top of mnist_image_node.

        Returns
        -------
        rval: list
          [Conv2DLayer, SoftmaxLayer]. This isn't all the nodes from
          input to output (it omits the scaling and formatting nodes).
        """

        assert_equal(str(sl_input_node.output_symbol.dtype), 'float32')
        assert_equal(sl_input_node.output_format.axes,
                     ('b', 'c', '0', '1'))

        cast = numpy.cast[floatX]

        rng = numpy.random.RandomState(seed)
        layers = []

        image_dims = scaled_image_node.output_format.shape[1:]

        last_node = sl_input_node
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
        filters.set_value(cast(rng.uniform(low=-conv_uniform_range,
                                           high=conv_uniform_range,
                                           size=filters.get_value().shape)))

        # Transposes from bc01 to b01c before flattening to bf, as is done
        # in Pylearn2's Conv2DSpace._format_as_impl()
        last_node = SoftmaxLayer(last_node,
                                 DenseFormat(axes=('b', 'f'),
                                             shape=(-1, num_classes),
                                             dtype=None),
                                 input_to_bf_map={('0', '1', 'c'): 'f'})

        layers.append(last_node)

        weights = last_node.affine_node.linear_node.params
        weights.set_value(cast(rng.standard_normal(weights.get_value().shape) *
                               affine_stddev))

        return layers

    sl_layers = make_sl_model(sl_input_node)

    def get_sl_function(sl_layers):
        input_symbol = sl_input_node.output_symbol
        output_symbols = [layer.output_symbol for layer in sl_layers]
        return theano.function([input_symbol], output_symbols)

    sl_function = get_sl_function(sl_layers)

    def get_sl_conv_function():
        input_symbol = sl_input_node.output_symbol
        output_symbol = sl_layers[0].conv2d_node.output_symbol
        return theano.function([input_symbol],
                               output_symbol)

    sl_conv_function = get_sl_conv_function()

    def get_sl_conv_bias_function():
        input_symbol = sl_input_node.output_symbol
        output_symbol = sl_layers[0].bias_node.output_symbol
        return theano.function([input_symbol],
                               output_symbol)

    sl_conv_bias_function = get_sl_conv_bias_function()

    def get_sl_conv_pool_function():
        input_symbol = sl_input_node.output_symbol
        output_symbol = sl_layers[0].pool2d_node.output_symbol
        return theano.function([input_symbol],
                               output_symbol)

    sl_conv_pool_function = get_sl_conv_pool_function()

    def get_sl_softmax_linear_function():
        input_symbol = sl_input_node.output_symbol
        output_symbol = sl_layers[1].affine_node.linear_node.output_symbol
        return theano.function([input_symbol],
                               output_symbol)

    sl_softmax_linear_function = get_sl_softmax_linear_function()

    def get_sl_softmax_bias_function():
        input_symbol = sl_input_node.output_symbol
        output_symbol = sl_layers[1].affine_node.bias_node.output_symbol
        return theano.function([input_symbol],
                               output_symbol)

    sl_softmax_bias_function = get_sl_softmax_bias_function()

    # def get_sl_softmax_softmax_function():
    #     input_symbol = sl_input_node.output_symbol
    #     output_symbol = sl_layers[1].softmax_node.output_symbol
    #     return theano.function([input_symbol],
    #                            output_symbol)

    # sl_softmax_relu_function = get_sl_softmax_relu_function()

    def make_pl_model(pl_input_node):
        '''
        Returns a pylearn2.models.mlp.MLP.

        This MLP expects a float32 image as input. See pylearn2's mnist.py,
        which calls pylearn2.utils.mnist_ubyte.read_mnist_images(path,
        'float32'), which rescales from [0, 255] to [0.0, 1.0].
        '''
        assert_equal(pl_input_node.output_format.axes,
                     ('b', '0', '1', 'c'))
        assert_equal(pl_input_node.output_symbol.dtype, 'float32')

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

        image_shape = pl_input_node.output_format.shape[1:3]

        result = mlp.MLP(
            layers=layers,
            seed=seed,
            batch_size=batch_size,
            input_space=pylearn2.space.Conv2DSpace(shape=image_shape,
                                                   num_channels=1))

        weights, biases = layers[0].get_params()
        sl_weights = sl_layers[0].conv2d_node.filters
        assert_array_equal(weights.get_value(), sl_weights.get_value())
        sl_biases = sl_layers[0].bias_node.params
        assert_equal(sl_biases.get_value().shape, (1, 2))
        assert_array_equal(biases.get_value(), sl_biases.get_value()[0])

        biases, weights = layers[1].get_params()
        sl_weights = sl_layers[1].affine_node.linear_node.params
        assert_array_equal(weights.get_value(), sl_weights.get_value())
        sl_biases = sl_layers[1].affine_node.bias_node.params
        assert_equal(sl_biases.get_value().shape, (1, num_classes))
        assert_array_equal(biases.get_value(), sl_biases.get_value()[0])

        return result

    pl_mlp = make_pl_model(pl_input_node)

    def get_pl_function(pl_mlp):
        '''
        Returns a compiled function that takes an floatX image batch
        and returns outputs from each of the MLP's layers.
        '''
        input_state = pl_input_node.output_symbol
        outputs = pl_mlp.fprop(state_below=input_state,
                               return_all=True)
        return theano.function([input_state], outputs)

    pl_function = get_pl_function(pl_mlp)

    def get_pl_conv_function(pl_mlp):
        input_state = pl_input_node.output_symbol
        output = pl_mlp.layers[0].DEBUG_conv_output
        return theano.function([input_state], output)

    pl_conv_function = get_pl_conv_function(pl_mlp)

    def get_pl_conv_bias_function(pl_mlp):
        input_state = pl_input_node.output_symbol
        output = pl_mlp.layers[0].DEBUG_bias_output
        return theano.function([input_state], output)

    pl_conv_bias_function = get_pl_conv_bias_function(pl_mlp)

    def get_pl_conv_pool_function(pl_mlp):
        input_state = pl_input_node.output_symbol
        output = pl_mlp.layers[0].DEBUG_pool_output
        return theano.function([input_state], output)

    pl_conv_pool_function = get_pl_conv_pool_function(pl_mlp)

    def get_pl_softmax_linear_function(pl_mlp):
        input_state = pl_input_node.output_symbol
        output = pl_mlp.layers[1].DEBUG_linear_output
        return theano.function([input_state], output)

    pl_softmax_linear_function = get_pl_softmax_linear_function(pl_mlp)

    def get_pl_softmax_bias_function(pl_mlp):
        input_state = pl_input_node.output_symbol
        output = pl_mlp.layers[1].DEBUG_bias_output
        return theano.function([input_state], output)

    pl_softmax_bias_function = get_pl_softmax_bias_function(pl_mlp)

    get_sl_input = theano.function([mnist_image_node.output_symbol],
                                   sl_input_node.output_symbol)

    get_pl_input = theano.function([mnist_image_node.output_symbol],
                                   pl_input_node.output_symbol)

    mnist_image_batch = mnist_image_node.output_format.make_batch(
        batch_size=batch_size,
        is_symbolic=False)

    image_rng = numpy.random.RandomState(2352)
    mnist_image_batch[...] = image_rng.random_integers(
        255,
        size=mnist_image_batch.shape)

    sl_input_batch = get_sl_input(mnist_image_batch)

    sl_outputs = sl_function(sl_input_batch)
    sl_conv_outputs = sl_conv_function(sl_input_batch)
    sl_conv_bias_outputs = sl_conv_bias_function(sl_input_batch)
    sl_conv_pool_outputs = sl_conv_pool_function(sl_input_batch)

    pl_input_batch = get_pl_input(mnist_image_batch)

    pl_outputs = pl_function(pl_input_batch)
    pl_conv_outputs = pl_conv_function(pl_input_batch)
    pl_conv_bias_outputs = pl_conv_bias_function(pl_input_batch)
    pl_conv_pool_outputs = pl_conv_pool_function(pl_input_batch)


    assert_array_equal(sl_input_batch, pl_input_batch.transpose((0, 3, 1, 2)))

    assert_array_equal(sl_conv_outputs, pl_conv_outputs)

    assert_array_equal(sl_conv_bias_outputs, pl_conv_bias_outputs)

    assert_array_equal(sl_conv_pool_outputs, pl_conv_pool_outputs)

    assert_array_equal(sl_outputs[0], pl_outputs[0])

    # Done checking conv layer

    sl_softmax_linear_output = sl_softmax_linear_function(sl_input_batch)
    sl_softmax_bias_output = sl_softmax_linear_function(sl_input_batch)

    pl_softmax_linear_output = pl_softmax_linear_function(pl_input_batch)
    pl_softmax_bias_output = pl_softmax_linear_function(pl_input_batch)

    # pdb.set_trace()
    assert_array_equal(sl_softmax_linear_output, pl_softmax_linear_output)
    assert_array_equal(sl_softmax_bias_output, pl_softmax_bias_output)
    assert_array_equal(sl_outputs[1], pl_outputs[1])

    # for sl_output, pl_output in safe_izip(sl_outputs, pl_outputs):
    #     assert_array_equal(sl_output, pl_output)

    print("All outputs are equal.")

if __name__ == '__main__':
    main()
