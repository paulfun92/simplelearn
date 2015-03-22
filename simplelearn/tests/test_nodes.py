'''
Tests for simplelearn.nodes
'''

import itertools
import numpy
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from numpy.testing import assert_allclose, assert_array_equal
from nose.tools import (assert_is_instance,
                        assert_equal,
                        assert_greater,
                        assert_greater_equal,
                        assert_true)
from simplelearn.formats import DenseFormat
from simplelearn.utils import (safe_izip,
                               assert_all_greater,
                               assert_all_greater_equal,
                               assert_all_less_equal)
from simplelearn.nodes import (Node,
                               InputNode,
                               Linear,
                               Bias,
                               Function1dTo1d,
                               Pool2D,
                               Conv2D,
                               ReLU,
                               Dropout,
                               Softmax,
                               L2Loss,
                               CrossEntropy,
                               _assert_is_shape2d)

from unittest import TestCase

import pdb


def _make_random_batch(rng, fmt, batch_size):
    batch = fmt.make_batch(is_symbolic=False, batch_size=batch_size)
    batch[...] = rng.uniform(size=batch.shape)
    return batch


class DummyFunction1dTo1d(Function1dTo1d):
    def __init__(self, output_format, input_node):
        super(DummyFunction1dTo1d, self).__init__(input_node, output_format)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        input_format = input_bf_node.output_format
        assert not input_format.requires_conversion(output_bf_format)

        return input_bf_node


class Function1dTo1dTester(TestCase):
    '''
    Subclass from this to test subclasses of Function1dTo1d.
    '''

    def setUp(self):
        input_format = DenseFormat(axes=('0', 'b', '1'),
                                   shape=(3, -1, 4),
                                   dtype=theano.config.floatX)

        self.input_node = InputNode(input_format)

        output_format = DenseFormat(axes=('k', 'b', 'f'),
                                    shape=(6, -1, 2),
                                    dtype=None)

        self.node = self._make_node(self.input_node, output_format)
        assert_is_instance(self.node, Node)

        # kwargs to feed to assert_allclose (e.g. rtol=0.001).
        # Change in subclasses' setUp methods (don't forget to call
        # this superclass' setUp, though)
        self.allclose_kwargs = {}

    def _make_node(self, input_node, output_format):
        return DummyFunction1dTo1d(output_format, input_node)

    def expected_function1dTo1d(self, rows):
        return rows

    def expected_function(self, input_batch):
        input_format = self.input_node.output_format
        input_batch = input_batch.transpose([input_format.axes.index(a)
                                             for a in ('b', '0', '1')])
        input_mat = input_batch.reshape((input_batch.shape[0],
                                         numpy.prod(input_batch.shape[1:])))

        output_mat = self.expected_function1dTo1d(input_mat)

        output_format = self.node.output_format
        non_b_axes = [a for a in output_format.axes if a != 'b']
        non_b_shape = \
            [self.node.output_format.shape[output_format.axes.index(a)]
             for a in non_b_axes]

        output_batch = output_mat.reshape([-1] + non_b_shape)
        output_axes = tuple(['b'] + non_b_axes)
        output_batch = output_batch.transpose([output_axes.index(a)
                                               for a in output_format.axes])

        return output_batch

    def test(self):
        rng = numpy.random.RandomState(14141)

        node_function = theano.function([self.input_node.output_symbol],
                                        self.node.output_symbol)

        batch_size = 5

        input_format = self.input_node.output_format

        for _ in range(3):
            input_batch = _make_random_batch(rng, input_format, batch_size)
            output_batch = node_function(input_batch)
            expected_output_batch = self.expected_function(input_batch)

            assert_allclose(output_batch,
                            expected_output_batch,
                            **self.allclose_kwargs)


class LinearTester(Function1dTo1dTester):

    def _make_node(self, input_node, output_format):
        return Linear(input_node, output_format)

    def expected_function1dTo1d(self, rows):
        return numpy.dot(rows, self.node.params.get_value())


class BiasTester(Function1dTo1dTester):
    def _make_node(self, input_node, output_format):
        return Bias(input_node, output_format)

    def expected_function1dTo1d(self, rows):
        params = self.node.params.get_value()
        assert_equal(params.shape[0], 1)
        assert_equal(rows.shape[1], params.shape[1])

        return rows + self.node.params.get_value()


class SoftmaxTester(Function1dTo1dTester):
    def setUp(self):
        super(SoftmaxTester, self).setUp()
        self.allclose_kwargs = {'rtol': 1e-6}

    def _make_node(self, input_node, output_format):
        return Softmax(input_node, output_format)

    def expected_function1dTo1d(self, rows):
        # For numerical stability.
        # Equiv. to dividing numerator and denominator by max exponent.
        rows -= numpy.max(rows, axis=1, keepdims=True)
        exp_rows = numpy.exp(rows)
        denominators = numpy.sum(exp_rows, axis=1, keepdims=True)
        return exp_rows / denominators


def test_ReLU():
    rng = numpy.random.RandomState(234)
    input_format = DenseFormat(axes=('b', '0', '1', 'c'),
                               shape=(-1, 2, 3, 4),
                               dtype='floatX')

    input_node = InputNode(fmt=input_format)
    relu = ReLU(input_node)
    relu_function = theano.function([input_node.output_symbol],
                                    relu.output_symbol)

    for batch_size in (1, 5):
        input_batch = _make_random_batch(rng, input_format, batch_size)
        output_batch = relu_function(input_batch)

        expected_output_batch = numpy.copy(output_batch)
        expected_output_batch[expected_output_batch < 0.0] = 0.0

        assert_array_equal(output_batch, expected_output_batch)


def test_dropout():
    rng = numpy.random.RandomState(4352)
    theano_rng = RandomStreams(8482)

    input_format = DenseFormat(axes=('k', 'd', 'b', 'f'),
                               shape=(20, 1, -1, 30),
                               dtype='floatX')

    input_node = InputNode(fmt=input_format)

    include_probability = .7
    dropout = Dropout(input_node, include_probability, theano_rng)

    dropout_and_mask_function = theano.function([input_node.output_symbol],
                                                [dropout.output_symbol,
                                                 dropout.mask])

    def get_actual_and_expected_values(input_batch):
        actual, mask = dropout_and_mask_function(input_batch)
        scale = numpy.cast[input_format.dtype](1.0 / include_probability)
        expected = input_batch * mask * scale

        return actual, expected

    for batch_size in range(3):
        input_batch = _make_random_batch(rng, input_format, batch_size)
        input_batch[input_batch == 0.0] = 0.1  # no zeros in input batch

        actual, expected = get_actual_and_expected_values(input_batch)
        assert_allclose(actual, expected)

        if batch_size > 0:
            num_included = numpy.count_nonzero(actual != 0.0)
            actual_include_fraction = (float(num_included) /
                                       float(input_batch.size))
            assert_allclose(actual_include_fraction,
                            include_probability,
                            atol=.05)


def test_l2loss():
    rng = numpy.random.RandomState(3523)

    def expected_loss_function(arg0, arg1, arg_format):
        diff = arg0 - arg1
        feature_size = numpy.prod([size for size, axis in
                                   safe_izip(arg_format.shape, arg_format.axes)
                                   if axis != 'b'])
        bf_format = DenseFormat(axes=('b', 'f'),
                                shape=(-1, feature_size),
                                dtype=None)
        non_b_axes = tuple(axis for axis in arg_format.axes if axis != 'b')
        axis_map = {non_b_axes : 'f'}
        diff = arg_format.convert(diff, bf_format, axis_map=axis_map)

        return (diff * diff).sum(axis=1)


    input_format = DenseFormat(axes=('c', '0', 'b', '1'),
                               shape=(1, 2, -1, 3),
                               dtype='floatX')
    input_node_a = InputNode(fmt=input_format)
    input_node_b = InputNode(fmt=input_format)
    loss_node = L2Loss(input_node_a, input_node_b)

    loss_function = theano.function([input_node_a.output_symbol,
                                     input_node_b.output_symbol],
                                    loss_node.output_symbol)

    for batch_size in xrange(4):
        batch_a = _make_random_batch(rng, input_format, batch_size)
        batch_b = _make_random_batch(rng, input_format, batch_size)

        actual_loss = loss_function(batch_a, batch_b)
        expected_loss = expected_loss_function(batch_a, batch_b, input_format)

        assert_equal(actual_loss.shape, expected_loss.shape)
        assert_allclose(actual_loss, expected_loss, rtol=1e-6)


def test_crossentropy():
    rng = numpy.random.RandomState(9239)

    def expected_loss_function(softmaxes, targets):
        def make_onehots(hot_indices):
            assert_true(numpy.issubdtype(hot_indices.dtype, numpy.integer))
            result = numpy.zeros(softmaxes.shape, dtype=softmaxes.dtype)
            result[xrange(result.shape[0]), hot_indices] = 1.0

            return result

        if targets.ndim == 1:
            targets = make_onehots(targets)

        return -numpy.sum(numpy.log(softmaxes) * targets, axis=1)

    def make_loss_function(softmax_format, target_format):
        softmax_node = InputNode(fmt=softmax_format)
        target_node = InputNode(fmt=target_format)
        cross_entropy = CrossEntropy(softmax_node, target_node)
        return theano.function([x.output_symbol for x in (softmax_node,
                                                          target_node)],
                        cross_entropy.output_symbol)



    for use_integer_targets in (True, False):
        for vec_size in (1, 2):
            if use_integer_targets:
                target_format = DenseFormat(axes=['b'],
                                            shape=[-1],
                                            dtype='int')
            else:
                target_format = DenseFormat(axes=['b', 'f'],
                                            shape=(-1, vec_size),
                                            dtype='int')

            softmax_format = DenseFormat(axes=('b', 'f'),
                                         shape=(-1, vec_size),
                                         dtype='floatX')


            loss_function = make_loss_function(softmax_format, target_format)

            for batch_size in range(3):
                softmaxes = _make_random_batch(rng, softmax_format, batch_size)
                targets = target_format.make_batch(batch_size=batch_size,
                                                   is_symbolic=False)

                hot_indices = rng.randint(0, vec_size, batch_size)

                if use_integer_targets:
                    targets[:] = hot_indices
                else:
                    targets[...] = 0.0
                    targets[xrange(targets.shape[0]), hot_indices] = 1.0

                actual_losses = loss_function(softmaxes, targets)
                expected_losses = expected_loss_function(softmaxes, targets)
                assert_allclose(actual_losses, expected_losses)


def _sliding_window_2d_testimpl(expected_subwindow_funcs,
                                make_node_funcs,
                                supports_padding):
    '''
    Implementation of tests for 2D sliding-window nodes like Pool2D and Conv2D.

    Parameters
    ----------
    expected_subwindow_funcs: Sequence
      A Sequence of subwindow functions.
      These take a subwindow and return a scalar.
      Input: tensor with shape [BATCH_SIZE, NUM_CHANNELS, ROWS, COLS]
      Output: tensor with shape [BATCH_SIZE, NUM_CHANNELS]

    make_node_funcs: Sequence
      A Sequence of functions that create sliding-window Nodes to be tested
      against the ground-truth provided by the corresponding
      expected_subwindow_funcs.

      Parameters
      ----------
      input_node: Node
      window_shape: Sequence
        [NUM_ROWS, NUM_COLUMNS] of the sliding window.
      strides: Sequence
        [ROW_STRIDE, COLUMN_STRIDE], or how many rows/columns to skip between
        applications of the sliding window.

      pad: Sequence
        [ROW_PAD, COLUMN_PAD], or # of zero-padding rows/columns to add to each
        side of the image.

      axis_map: dict
        Maps strings to strings. Optional.
        If the node uses different axis names than 'b', 'c', '0', '1', this
        specifies the mapping from the node's axis names to 'b', 'c', '0', '1'.

    supports_padding: bool
      True if the nodes being tested support zero-padding; otherwise False.
    '''

    def apply_subwindow_func(subwindow_func,
                             max_padded_images,
                             max_pad,
                             actual_pad,
                             window_shape,
                             strides):
        '''
        Applies a sliding-window function to all subwindows of a feature map.

        Parameters
        ----------
        subwindow_func: function
          A function that takes a subwindow and returns a scalar.
          Input: tensor with shape [BATCH_SIZE, NUM_CHANNELS, ROWS, COLS]
          Output: tensor with shape [BATCH_SIZE, NUM_CHANNELS]

        max_padded_images: numpy.ndarray
          A feature map with shape [BATCH_SIZE, NUM_CHANNELS, ROWS, COLS].
          This has max_pad[0] rows and max_pad[1] columns of zero-padding.

        max_pad: Sequence
          [pad_rows, pad_columns], the # of padded rows and columns on each
          side of the image.

        '''
        assert_equal(max_padded_images.ndim, 4)
        assert_all_less_equal(actual_pad, max_pad)
        assert_all_greater(max_padded_images.shape[2:], max_pad)
        _assert_is_shape2d(window_shape)
        _assert_is_shape2d(strides)

        max_pad, actual_pad, window_shape, strides = (numpy.asarray(a)
                                                      for a in (max_pad,
                                                                actual_pad,
                                                                window_shape,
                                                                strides))
        offsets = max_pad - actual_pad
        image_shape = numpy.asarray(max_padded_images.shape[2:]) - 2 * max_pad
        assert_all_greater(image_shape, 0)

        rows, cols = (range(offsets[i],
                            (offsets[i] +
                             image_shape[i] +
                             actual_pad[i] -
                             window_shape[i] +
                             1),
                            strides[i])
                      for i in (0, 1))
        output_image = None

        for out_r, in_r in enumerate(rows):
            for out_c, in_c in enumerate(cols):
                subwindow = max_padded_images[:,
                                              :,
                                              in_r:(in_r + window_shape[0]),
                                              in_c:(in_c + window_shape[1])]
                output = subwindow_func(subwindow)
                assert_equal(output.ndim, 2)

                # check that subwindow_func preserved the batch size
                assert_equal(output.shape[0], max_padded_images.shape[0])
                assert_greater(output.shape[1], 0)

                if output_image is None:
                    output_image = numpy.zeros((output.shape[0],
                                                output.shape[1],
                                                len(rows),
                                                len(cols)),
                                               dtype=output.dtype)

                output_image[:, :, out_r, out_c] = output

        return output_image

    # supports_padding = False # TODO: make this an arg, set to Falseonly for test_pool2d

    max_stride = 2  # 3?
    max_window_size = 3 # next: 2, then finally 3
    batch_size = 2
    num_channels = 2
    input_dtype = numpy.dtype('int')

    if supports_padding:
        max_pad = 3
        assert_greater(max_pad, max_window_size)
    else:
        max_pad = 0

    assert_greater_equal(max_pad, 0)

    rng = numpy.random.RandomState(352)

    max_padded_images = numpy.zeros((batch_size,
                                     num_channels,
                                     max_pad*2 + max_window_size + 1,
                                     max_pad*2 + max_window_size + 4),
                                    dtype=input_dtype)
    if max_pad > 0:
        images = max_padded_images[:, :, max_pad:-max_pad, max_pad:-max_pad]
    else:
        images = max_padded_images[...]

    images[...] = rng.random_integers(low=-10, high=10, size=images.shape)

    axis_map = None  # TODO: rename axes (to 'bee' 'zero' 'one' 'see')
    # TODO: rename as above, shuffle axes
    input_node = InputNode(DenseFormat(axes=('b', 'c', '0', '1'),
                                       shape=(-1, ) + images.shape[1:],
                                       dtype=input_dtype))
    images = images.transpose([input_node.output_format.axes.index(a)
                               for a in ('b', 'c', '0', '1')])  # TODO: rename but keep order
    assert_all_greater(images.shape, 0)

    prod = itertools.product

    for expected_func, make_node_func in safe_izip(expected_subwindow_funcs,
                                                   make_node_funcs):

        # Loops through all possible window_shapes, pads, and strides
        for window_shape in prod(range(1, max_window_size + 1), repeat=2):
            for pads in prod(range(max_pad + 1), repeat=2):
                for strides in prod(range(1, max_stride + 1), repeat=2):
                    node = make_node_func(input_node,
                                          window_shape=window_shape,
                                          strides=strides,
                                          pads=pads,
                                          axis_map=axis_map)

                    node_func = theano.function([input_node.output_symbol],
                                                node.output_symbol)

                    expected_images = apply_subwindow_func(expected_func,
                                                           max_padded_images,
                                                           max_pad,
                                                           pads,
                                                           window_shape,
                                                           strides)

                    actual_images = node_func(images)

                    assert_allclose(actual_images, expected_images)


def test_pool2d():
    def average_pool(subwindow):
        assert_equal(subwindow.ndim, 4)

        num_pixels = numpy.prod(subwindow.shape[2:])
        assert_greater(num_pixels, 0)
        subwindow = subwindow.reshape((subwindow.shape[0],
                                       subwindow.shape[1],
                                       num_pixels))
        return subwindow.sum(axis=-1) / float(num_pixels)

    def max_pool(subwindow):
        assert_equal(subwindow.ndim, 4)

        subwindow = subwindow.reshape((subwindow.shape[0],
                                       subwindow.shape[1],
                                       -1))
        return subwindow.max(axis=-1)

    def make_average_pool_node(input_node,
                               window_shape,
                               strides,
                               pads,  # ignored
                               axis_map):
        return Pool2D(input_node=input_node,
                      window_shape=window_shape,
                      strides=strides,
                      mode='average',
                      axis_map=axis_map)

    def make_max_pool_node(input_node,
                           window_shape,
                           strides,
                           pads,  # ignored
                           axis_map):
        return Pool2D(input_node=input_node,
                      window_shape=window_shape,
                      strides=strides,
                      mode='max',
                      axis_map=axis_map)

    # make_average_pool_node, make_max_pool_node = (
    #     lambda input_node, window_shape, strides, pad, axis_map: \
    #     return Pool2D(input_node=input_node,
    #                   window_shape=window_shape,
    #                   strides=strides,
    #                   mode=mode,
    #                   axis_map=axis_map)
    #     for mode in ('average', 'max'))

    # def make_maxpool_node(input_node,
    #                       window_shape,
    #                       strides,
    #                       pad,  # ignored
    #                       axis_map):
    #     return Pool2D(input_node=input_node,
    #                   window_shape=window_shape,
    #                   strides=strides,
    #                   mode=mode,
    #                   axis_map=axis_map)

    # def make_node(input_node,
    #               window_shape,
    #               strides,
    #               pad,  # ignored
    #               mode,
    #               axis_map):
    #     return Pool2D(input_node=input_node,
    #                   window_shape=window_shape,
    #                   strides=strides,
    #                   mode=mode,
    #                   axis_map=axis_map)

    _sliding_window_2d_testimpl([average_pool, max_pool],
                                [make_average_pool_node, make_max_pool_node],
                                supports_padding=False)
