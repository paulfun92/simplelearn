'''
Tests for simplelearn.nodes
'''

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2015"
__license__ = "Apache 2.0"


from collections import Sequence
import itertools
import numpy
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from numpy.testing import assert_allclose, assert_array_equal
from nose.tools import (assert_is_instance,
                        assert_equal,
                        assert_greater,
                        assert_greater_equal,
                        assert_true,
                        assert_raises_regexp)
from simplelearn.formats import DenseFormat
from simplelearn.utils import safe_izip, cudnn_available
from simplelearn.asserts import assert_all_greater
from simplelearn.nodes import (Node,
                               FormatNode,
                               InputNode,
                               Linear,
                               Bias,
                               AffineTransform,
                               Function1dTo1d,
                               Pool2D,
                               CuDnnConv2d,
                               Conv2d,
                               ReLU,
                               Dropout,
                               Softmax,
                               L2Loss,
                               CrossEntropy,
                               Lcn,
                               Conv2dLayer,
                               _assert_is_shape2d,
                               _make_2d_gaussian_filter)

from unittest import TestCase

pylearn2_installed = True
try:
    from pylearn2.models import mlp
except ImportError:
    pylearn2_installed = False

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
                            input_bf_format,
                            output_bf_format):
        input_format = input_bf_node.output_format
        assert not input_format.requires_conversion(output_bf_format)

        return input_bf_node


def test_format_node():
    '''
    A simple test for FormatNode.
    '''
    input_node = InputNode(fmt=DenseFormat(axes=('b', 'zero', 'see', 'one'),
                                           shape=(-1, 4, 3, 5),
                                           dtype=int))

    output_format = DenseFormat(axes=('b', 'c', '0', '1'),
                                shape=(-1, 3, 4, 5),
                                dtype=float)

    format_node = FormatNode(input_node, output_format, {'zero': '0',
                                                         'see': 'c',
                                                         'one': '1'})

    format_func = theano.function([input_node.output_symbol],
                                  format_node.output_symbol)

    def format_batch(batch):
        batch = batch.transpose(0, 2, 1, 3)
        if output_format.dtype is None:
            return batch
        else:
            return numpy.cast[output_format.dtype](batch)

    batch = input_node.output_format.make_batch(is_symbolic=False,
                                                batch_size=2)

    rng = numpy.random.RandomState(33215)
    batch[...] = rng.random_integers(low=-5, high=5, size=batch.shape)

    actual_formatted_batch = format_func(batch)
    expected_formatted_batch = format_batch(batch)

    assert_array_equal(actual_formatted_batch, expected_formatted_batch)


class Function1dTo1dTester(TestCase):
    '''
    Subclass from this to test subclasses of Function1dTo1d.
    '''

    def setUp(self):
        input_format = DenseFormat(axes=('0', 'b', '1'),
                                   shape=(3, -1, 4),
                                   # pylint: disable=no-member
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
        assert_equal(params.ndim, 1)
        assert_equal(rows.shape[1], params.shape[0])

        return rows + self.node.params.get_value()


class AffineTester(Function1dTo1dTester):
    def _make_node(self, input_node, output_format):
        return AffineTransform(input_node, output_format)

    def expected_function1dTo1d(self, rows):
        return (numpy.dot(rows, self.node.linear_node.params.get_value()) +
                self.node.bias_node.params.get_value())


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
        axis_map = {non_b_axes: 'f'}
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

    for batch_size in xrange(1, 4):
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
                assert_allclose(actual_losses, expected_losses, atol=1e-6)


def _sliding_window_2d_testimpl(expected_subwindow_funcs,
                                pad_values,
                                make_node_funcs,
                                make_pad_args_funcs,
                                rtol=None):
    '''
    Implementation of tests for 2D sliding-window nodes like Pool2D and Conv2d.

    Parameters
    ----------
    expected_subwindow_funcs: Sequence
      A Sequence of subwindow functions.
      These take a subwindow and return a scalar.
      Input: tensor with shape [BATCH_SIZE, NUM_CHANNELS, ROWS, COLS]
      Output: tensor with shape [BATCH_SIZE, NUM_CHANNELS]

    pad_values: Sequence
      A sequence of pad filler values to use for eah of the
      expected_subwindow_funcs. For example, if expected_subwindow_funcs
      is [average_pool, max_pool], use [0.0, -numpy.inf].

    make_node_funcs: Sequence
      A Sequence of functions that create sliding-window Nodes to be tested
      against the ground-truth provided by the corresponding
      expected_subwindow_funcs. Its paramters are as follows:

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

    make_pad_args_funcs: Sequence
      A Sequence of functions that take a window_shape arg (2d array) and
      returns an Iterable of 'pad' arguments, which can be strings or 2d arrays
      of ints.
    '''

    assert_is_instance(expected_subwindow_funcs, Sequence)
    assert_is_instance(pad_values, Sequence)
    assert_is_instance(make_node_funcs, Sequence)

    # TODO: change this to construct a Toeplitz matrix out of padded_images,
    # so we get a giant stack of C X WR X WC matrices, which can then be fed
    # to subwindow_func as a single batch.
    # See scipy.linalg.toeplitz
    def apply_subwindow_func(subwindow_func,
                             padded_images,
                             pads,
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

        padded_images: numpy.ndarray
          A feature map with shape [BATCH_SIZE, NUM_CHANNELS, ROWS, COLS].
          This has pad[0] rows and pad[1] columns of zero-padding.

        max_pad: Sequence
          [pad_rows, pad_columns], the # of padded rows and columns on each
          side of the image.

        '''
        assert_equal(padded_images.ndim, 4)
        assert_all_greater(padded_images.shape[2:], pads)
        _assert_is_shape2d(window_shape)
        _assert_is_shape2d(strides)

        pads, window_shape, strides = (numpy.asarray(a) for a in (pads,
                                                                  window_shape,
                                                                  strides))

        assert_all_greater(numpy.asarray(padded_images.shape[2:]), 2 * pads)

        # Check that pad region is full of the same value
        if pads[0] > 0:
            pad_value = padded_images[0, 0, 0, 0]
            assert_true(numpy.all(padded_images[:, :, :pads[0], :] ==
                                  pad_value))
            assert_true(numpy.all(padded_images[:, :, -pads[0]:, :] ==
                                  pad_value))

        if pads[1] > 0:
            pad_value = padded_images[0, 0, 0, 0]
            assert_true(numpy.all(padded_images[:, :, :, :pads[1]] ==
                                  pad_value))
            assert_true(numpy.all(padded_images[:, :, :, -pads[1]:] ==
                                  pad_value))

        rows, cols = (range(0,
                            padded_images.shape[i + 2] - window_shape[i] + 1,
                            strides[i])
                      for i in (0, 1))
        output_image = None

        for out_r, in_r in enumerate(rows):
            for out_c, in_c in enumerate(cols):
                subwindow = padded_images[:,
                                          :,
                                          in_r:(in_r + window_shape[0]),
                                          in_c:(in_c + window_shape[1])]
                output = subwindow_func(subwindow)
                assert_equal(output.ndim, 2)

                # check that subwindow_func preserved the batch size
                assert_equal(output.shape[0], padded_images.shape[0])
                assert_greater(output.shape[1], 0)

                if output_image is None:
                    output_image = numpy.zeros((output.shape[0],
                                                output.shape[1],
                                                len(rows),
                                                len(cols)),
                                               dtype=output.dtype)

                output_image[:, :, out_r, out_c] = output

        return output_image

    max_stride = 3
    max_window_size = 3
    batch_size = 2
    num_channels = 2
    input_dtype = numpy.dtype('int')

    max_pad = max_window_size + 1

    assert_greater_equal(max_pad, 0)

    rng = numpy.random.RandomState(352)

    def get_padded_image(max_padded_images, pads):
        def margin_to_slice(margin):
            assert_greater_equal(margin, 0)
            if margin == 0:
                return slice(None, None)
            else:
                return slice(margin, -margin)

        return max_padded_images[:,
                                 :,
                                 margin_to_slice(max_pad - pads[0]),
                                 margin_to_slice(max_pad - pads[1])]


    def get_pads_from_pad_arg(pad_arg, window_shape):
        '''
        Converts a valid pad argument (str or 2-int Sequence) to
        an equivalent 2-int numpy.ndarray.
        '''
        window_shape = numpy.asarray(window_shape)
        _assert_is_shape2d(window_shape)

        if isinstance(pad_arg, basestring):
            if pad_arg == 'full':
                return window_shape - 1
            elif pad_arg == 'valid':
                return numpy.asarray([0, 0])
            elif pad_arg == 'same_shape':
                assert_true((window_shape % 2 != 0).all())
                return window_shape // 2
            else:
                raise ValueError("Unrecognized pad name: '%s'" % pad_arg)
        else:
            _assert_is_shape2d(pad_arg)
            return numpy.asarray(pad_arg)


    prod = itertools.product

    for (expected_func,
         pad_value,
         make_node_func,
         make_pad_args_func) in safe_izip(expected_subwindow_funcs,
                                          pad_values,
                                          make_node_funcs,
                                          make_pad_args_funcs):

        # An image with the maximum amount of padding.  We will vary the amount
        # of padding in practice by taking centered subwindows of this image.
        max_padded_images = numpy.empty((batch_size,
                                         num_channels,
                                         max_pad * 2 + max_window_size + 1,
                                         max_pad * 2 + max_window_size + 4),
                                        dtype=input_dtype)
        max_padded_images[...] = pad_value

        images = get_padded_image(max_padded_images, (0, 0))
        images[...] = rng.random_integers(low=-10, high=10, size=images.shape)
        #images[...] = numpy.arange(images.size).reshape(images.shape)
        assert_all_greater(images.shape, 0)

        if max_pad == 0:
            assert_array_equal(images, max_padded_images)
        else:
            assert_array_equal(images, max_padded_images[:,
                                                         :,
                                                         max_pad:-max_pad,
                                                         max_pad:-max_pad])

        # Make input_nodes with weird axis names and axis order
        axis_map = {'b': 'b', 'see': 'c', 'zero': '0', 'one': '1'}
        input_node_axes = ('b', 'see', 'zero', 'one')
        # input_node_axes = ('b', 'zero', 'see', 'one')
        transpose_indices = [('b', 'see', 'zero', 'one').index(a)
                             for a in input_node_axes]
        input_node_shape = [images.shape[t] for t in transpose_indices]
        input_node_shape[input_node_axes.index('b')] = -1
        input_node = InputNode(DenseFormat(axes=input_node_axes,
                                           shape=input_node_shape,
                                           dtype=input_dtype))

        # Loops through all possible window_shapes, pads (including padding
        # bigger than the window shape), strides.
        for window_shape in prod(range(1, max_window_size + 1), repeat=2):
            window_shape = numpy.asarray(window_shape)

            # for pad_arg in get_pad_args(window_shape, supports_padding):
            for pad_arg in make_pad_args_func(window_shape):
                # can't use same_shape padding with even window dims
                if pad_arg == 'same_shape' and (window_shape % 2 == 0).any():
                    continue

                pads = get_pads_from_pad_arg(pad_arg, window_shape)
                padded_images = get_padded_image(max_padded_images, pads)
                assert_array_equal(numpy.asarray(padded_images.shape[2:]),
                                   (2 * pads) +
                                   numpy.asarray(images.shape[2:]))

                for strides in prod(range(1, max_stride + 1), repeat=2):
                    expected_images = apply_subwindow_func(expected_func,
                                                           padded_images,
                                                           pads,
                                                           window_shape,
                                                           strides)

                    # If pads are bigger than window_size, expect an exception
                    # when creating the node.
                    if not isinstance(pads, basestring) and \
                       numpy.any(pads >= window_shape):
                        assert_raises_regexp(AssertionError,
                                             "Not all pads",
                                             make_node_func,
                                             input_node,
                                             window_shape=window_shape,
                                             strides=strides,
                                             pads=pad_arg,
                                             axis_map=axis_map)
                    else:
                        node = make_node_func(input_node,
                                              window_shape=window_shape,
                                              strides=strides,
                                              pads=pad_arg,
                                              axis_map=axis_map)

                        node_func = theano.function([input_node.output_symbol],
                                                    node.output_symbol)
                        transposed_images = images.transpose(transpose_indices)
                        actual_images = node_func(transposed_images)

                        node.output_format.check(actual_images)
                        # try:
                        #     node.output_format.check(actual_images)
                        # except AssertionError:
                        #     pdb.set_trace()

                        kwargs = {}
                        if rtol is not None:
                            kwargs['rtol'] = rtol

                        # pylint: disable=star-args
                        assert_allclose(actual_images,
                                        expected_images,
                                        **kwargs)
                        # try:
                        #     # pylint: disable=star-args
                        #     assert_allclose(actual_images,
                        #                     expected_images,
                        #                     **kwargs)
                        # except AssertionError:
                        #     pdb.set_trace()


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
                               pads,
                               axis_map):
        return Pool2D(input_node=input_node,
                      window_shape=window_shape,
                      strides=strides,
                      mode='average',
                      pad=pads,
                      axis_map=axis_map)

    def make_max_pool_node(input_node,
                           window_shape,
                           strides,
                           pads,
                           axis_map):
        return Pool2D(input_node=input_node,
                      window_shape=window_shape,
                      strides=strides,
                      mode='max',
                      pad=pads,
                      axis_map=axis_map)

    def get_pad_args(window_shape):
        return itertools.chain(('same_shape', 'full', 'valid'),
                               itertools.product(range(window_shape[0] + 1),
                                                 range(window_shape[1] + 1)))

    _sliding_window_2d_testimpl([average_pool, max_pool],
                                pad_values=[0.0, -numpy.inf],
                                make_node_funcs=[make_average_pool_node,
                                                 make_max_pool_node],
                                make_pad_args_funcs=[get_pad_args,
                                                     get_pad_args])


def test_pool2d_pylearn2():
    '''
    Tests that pool2d's 'pylearn2' padding mode indeed creates the same results
    as pylearn2's pooling operator.
    '''
    if not pylearn2_installed:
        return

    floatX = theano.config.floatX

    batch_size = 2
    num_channels = 2  # num. of conv filters
    image_shape = (2, 3)  # size of conv+bias output
    input_node = InputNode(DenseFormat(axes=('b', 'c', '0', '1'),
                                       shape=(-1, num_channels) + image_shape,
                                       dtype=floatX))

    pool_shape = (2, 2)
    pool_strides = (2, 2)
    pl_pooled_symbol = mlp.max_pool(bc01=input_node.output_symbol,
                                    pool_shape=pool_shape,
                                    pool_stride=pool_strides,
                                    image_shape=image_shape)

    pl_pool_func = theano.function([input_node.output_symbol],
                                   pl_pooled_symbol)

    sl_pool_node = Pool2D(input_node=input_node, window_shape=pool_shape,
                          strides=pool_strides, mode='max',
                          pad='pylearn2')

    sl_pool_func = theano.function([input_node.output_symbol],
                                   sl_pool_node.output_symbol)

    input_batch = numpy.arange(batch_size *
                               numpy.prod(input_node.output_format.shape[1:]))
    input_batch = numpy.cast[floatX](input_batch)
    input_batch = input_batch.reshape(input_node.output_format.shape)

    pl_pooled_batch = pl_pool_func(input_batch)
    sl_pooled_batch = sl_pool_func(input_batch)
    assert_array_equal(sl_pooled_batch, pl_pooled_batch)


def test_pool2d_quick():
    '''
    Tests max and avg pooling for all pad keyword values.
    Confirms that they pad with -inf and 0, respectively.
    '''
    # make all inputs negative, to make sure max pooling pads with -inf, while
    # avg pooling pads with 0.
    input_image = numpy.asarray([1, 2, 3, 4, 3, 2, 1], dtype='float32')
    input_image -= 5.0
    assert_true((input_image < 0).all())

    # convert to bc01 format
    newaxis = numpy.newaxis
    input_image = input_image[newaxis, newaxis, newaxis, :]

    def make_pool_func(mode, pad):
        shape = list(input_image.shape)
        shape[0] = -1
        input_node = InputNode(DenseFormat(shape=shape,
                                           axes=('b', 'c', '0', '1'),
                                           dtype=input_image.dtype))
        pool_node = Pool2D(input_node,
                           (1, 4),
                           (1, 2),
                           mode=mode,
                           pad=pad)
        return theano.function([input_node.output_symbol],
                               pool_node.output_symbol[0, 0, 0, :])

    max_pooled_pylearn2_padding = \
        make_pool_func(mode='max', pad='pylearn2')(input_image)
    assert_array_equal(max_pooled_pylearn2_padding, [-1, -1, -2])

    max_pooled_min_padding = make_pool_func(mode='max', pad='min')(input_image)
    assert_array_equal(max_pooled_min_padding, [-2, -1, -1])

    max_pooled_valid_padding = \
        make_pool_func(mode='max', pad='valid')(input_image)
    assert_array_equal(max_pooled_valid_padding, [-1, -1])

    max_pooled_full_padding = \
        make_pool_func(mode='max', pad='full')(input_image)
    assert_array_equal(max_pooled_full_padding, [-4, -2, -1, -1, -3])

    max_pooled_2_padding = make_pool_func(mode='max', pad=(0, 2))(input_image)
    assert_array_equal(max_pooled_2_padding, [-3, -1, -1, -2])

    avg_pooled_pylearn2_padding = \
        make_pool_func(mode='average', pad='pylearn2')(input_image)
    assert_array_equal(avg_pooled_pylearn2_padding, [-2.5, -2, -9 / 4.])

    avg_pooled_min_padding = \
        make_pool_func(mode='average', pad='min')(input_image)
    assert_allclose(avg_pooled_min_padding, [-2.25, -2, -2.5])

    avg_pooled_valid_padding = \
        make_pool_func(mode='average', pad='valid')(input_image)
    assert_allclose(avg_pooled_valid_padding, [-2.5, -2])

    avg_pooled_full_padding = \
        make_pool_func(mode='average', pad='full')(input_image)
    assert_allclose(avg_pooled_full_padding, [-1, -9 / 4., -2, -2.5, -7 / 4.])


def test_cudnn_conv2d():
    def rand_floats(shape):
        rng = numpy.random.RandomState(382342)
        return rng.uniform(low=-10, high=10, size=shape)

    num_filters = 10

    def cross_correlate(subwindow):
        '''
        Convolution without flipping the filters (aka cross-correlation).
        '''
        floatX = theano.config.floatX  # pylint: disable=no-member
        subwindow = numpy.cast[floatX](subwindow)
        filters = numpy.zeros(shape=(num_filters, ) + subwindow.shape[1:],
                              dtype=floatX)
        filters[...] = rand_floats(filters.shape)

        return numpy.einsum('bcde,fcde->bf', subwindow, filters)

    def make_cudnn_conv2d(input_node,
                          window_shape,
                          strides,
                          pads,
                          axis_map):
        result = CuDnnConv2d(input_node,
                             window_shape,
                             num_filters,
                             pads,
                             strides,
                             axis_map,
                             conv_mode='cross')
        filters = result.filters.get_value()
        filters[...] = rand_floats(filters.shape)
        result.filters.set_value(filters)

        return result

    def make_cudnn_conv2d_pad_args(window_shape):
        return itertools.chain(('same_shape', 'full', 'valid'),
                               itertools.product(range(window_shape[0] + 1),
                                                 range(window_shape[1] + 1)))


    _sliding_window_2d_testimpl(
        [cross_correlate],
        pad_values=[0.0],
        make_node_funcs=[make_cudnn_conv2d],
        make_pad_args_funcs=[make_cudnn_conv2d_pad_args],
        rtol=1e-3)

def test_conv2d():

    def rand_floats(shape):
        rng = numpy.random.RandomState(382342)
        return rng.uniform(low=-10, high=10, size=shape)

    num_filters = 10

    def cross_correlate(subwindow):
        '''
        Convolution without flipping the filters (aka cross-correlation).
        '''
        floatX = theano.config.floatX  # pylint: disable=no-member
        subwindow = numpy.cast[floatX](subwindow)
        filters = numpy.zeros(shape=(num_filters, ) + subwindow.shape[1:],
                              dtype=floatX)
        filters[...] = rand_floats(filters.shape)

        return numpy.einsum('bcde,fcde->bf', subwindow, filters)

    def convolve(subwindow):
        '''
        Convolution (filters are left-right flipped).
        '''
        floatX = theano.config.floatX  # pylint: disable=no-member
        subwindow = numpy.cast[floatX](subwindow)
        filters = numpy.zeros(shape=(num_filters, ) + subwindow.shape[1:],
                              dtype=floatX)
        filters[...] = rand_floats(filters.shape)

        flipped_filters = numpy.empty_like(filters)
        flipped_filters[...] = filters[:, :, ::-1, ::-1]

        return numpy.einsum('bcde,fcde->bf', subwindow, flipped_filters)

    def make_conv2d(input_node,
                    window_shape,
                    strides,
                    pads,
                    axis_map):
        result = Conv2d(input_node,
                        window_shape,
                        num_filters,
                        pads,
                        strides,
                        axis_map)
        filters = result.filters.get_value()
        filters[...] = rand_floats(filters.shape)
        result.filters.set_value(filters)

        return result

    def make_conv2d_pad_args(window_shape):
        return ('full', 'valid')

    if cudnn_available('conv'):
        conv_function = convolve
    else:
        conv_function = cross_correlate

    _sliding_window_2d_testimpl(
        [conv_function],
        pad_values=[0.0],
        make_node_funcs=[make_conv2d],
        make_pad_args_funcs=[make_conv2d_pad_args],
        rtol=1e-3)


def test_make_2d_gaussian_filter():
    dtype = theano.config.floatX  # pylint: disable=no-member

    # Test a range of filter shapes: square, non-square, even dims, odd dims
    for filter_shape in itertools.product(range(3, 7), repeat=2):
        filter_shape = numpy.asarray(filter_shape)

        def make_expected_filter(filter_shape):
            standard_deviations = filter_shape / 4.0

            covariance = numpy.diag(standard_deviations ** 2)
            inv_covariance = numpy.diag(1.0 / standard_deviations ** 2)

            mean = filter_shape // 2
            xys = numpy.indices(filter_shape)
            xys = xys.transpose((1, 2, 0)).reshape(numpy.prod(filter_shape), 2)

            rt = xys - mean[numpy.newaxis, :]
            rt_ic = numpy.dot(rt, inv_covariance)
            rt_ic_r = (rt_ic * rt).sum(axis=1)

            exp_term = numpy.exp(-0.5 * rt_ic_r)
            denominator = numpy.sqrt((2 * numpy.pi) ** 2 *
                                     numpy.linalg.det(covariance))
            return (exp_term / denominator).reshape(filter_shape)

        expected_filter = make_expected_filter(filter_shape)
        actual_filter = _make_2d_gaussian_filter(filter_shape, dtype=dtype)

        assert_allclose(actual_filter, expected_filter)


def test_lcn():
    '''
    This only tests whether LCN'ing a multi-channel image has the same effect
    as LCN-ing the same data as separate mono-channel images.

    To see if LCN 'looks' right, just run scripts/browse_norb.py with --lcn.
    '''

    rng = numpy.random.RandomState(48532)
    batch_size = 2
    num_channels = 3
    num_rows = 96
    num_columns = 108
    dtype = theano.config.floatX
    cast = numpy.cast[dtype]

    rgb_image_batch = cast(rng.uniform(size=(batch_size,
                                             num_channels,
                                             num_rows,
                                             num_columns)))

    rgb_format = DenseFormat(axes=('b', 'c', '0', '1'),
                             shape=(-1, num_channels, num_rows, num_columns),
                             dtype=dtype)

    rgb_node = InputNode(rgb_format)
    rgb_lcn = Lcn(rgb_node)
    rgb_function = theano.function([rgb_node.output_symbol],
                                   rgb_lcn.output_symbol)
    rgb_lcn_batch = rgb_function(rgb_image_batch)

    mono_image_batch = rgb_image_batch.reshape((batch_size * num_channels,
                                                1,
                                                num_rows,
                                                num_columns))

    mono_format = DenseFormat(axes=('b', 'c', '0', '1'),
                              shape=(-1, 1, num_rows, num_columns),
                              dtype=dtype)

    mono_node = InputNode(mono_format)
    mono_lcn = Lcn(mono_node)
    mono_function = theano.function([mono_node.output_symbol],
                                    mono_lcn.output_symbol)
    mono_lcn_batch = mono_function(mono_image_batch)

    assert_allclose(rgb_lcn_batch.reshape((-1, 1, num_rows, num_columns)),
                    mono_lcn_batch)


def test_conv_layer():
    '''
    Sees if the Conv2dLayer properly implements
    conv -> channel-wise bias -> relu -> max pool
    '''

    # this b01c ordering will get reordered to bc01
    image_format = DenseFormat(axes=('b', '0', '1', 'c'),
                               shape=(-1, 3, 10, 12),
                               dtype=theano.config.floatX)

    image_node = InputNode(fmt=image_format)

    rng_seed = 13515

    # Conv layer parameters
    filter_shape = (3, 3)
    num_filters = 2
    conv_pads = 'same_shape'
    pool_shape = (4, 4)
    pool_strides = (2, 2)

    def randomize(rng, params):
        values = params.get_value()
        values[...] = rng.uniform(low=-.05, high=.05, size=values.shape)
        params.set_value(values)

    def make_conv_layer():

        conv_layer_node = Conv2dLayer(image_node,
                                      filter_shape=filter_shape,
                                      num_filters=num_filters,
                                      conv_pads=conv_pads,
                                      pool_window_shape=pool_shape,
                                      pool_strides=pool_strides)

        filters = conv_layer_node.conv2d_node.filters
        biases = conv_layer_node.bias_node.params

        rng = numpy.random.RandomState(rng_seed)

        randomize(rng, filters)
        randomize(rng, biases)
        return conv_layer_node

    conv_layer_node = make_conv_layer()
    conv_layer_function = theano.function([image_node.output_symbol],
                                          conv_layer_node.output_symbol)

    #
    # Construct a chain of lesser nodes that should have the same effect as
    # conv_layer_node. In particular, use a more direct way of expressing the
    # bias.
    #

    def make_conv_sequence():
        Conv2dNodeClass = (CuDnnConv2d if theano.sandbox.cuda.dnn.dnn_available
                           else Conv2d)

        conv2d_node = Conv2dNodeClass(image_node,
                                      filter_shape=filter_shape,
                                      num_filters=num_filters,
                                      pads=conv_pads)

        biases = theano.shared(numpy.zeros((1, num_filters, 1, 1),
                                           dtype=theano.config.floatX),
                               broadcastable=(True, False, True, True))

        bias_node = Node(conv2d_node,
                         conv2d_node.output_symbol + biases,
                         conv2d_node.output_format)

        relu_node = ReLU(bias_node)
        pool_node = Pool2D(relu_node,
                           mode='max',
                           window_shape=pool_shape,
                           strides=pool_strides,
                           pad='valid')

        rng = numpy.random.RandomState(rng_seed)
        randomize(rng, conv2d_node.filters)
        randomize(rng, biases)

        return pool_node

    conv_sequence_output_node = make_conv_sequence()
    conv_sequence_function = theano.function(
        [image_node.output_symbol],
        conv_sequence_output_node.output_symbol)

    batch_size = 3
    image = image_node.output_format.make_batch(is_symbolic=False,
                                                batch_size=batch_size)

    actual_output = conv_layer_function(image)
    expected_output = conv_sequence_function(image)

    assert_allclose(actual_output, expected_output)
