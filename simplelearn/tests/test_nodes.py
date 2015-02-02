'''
Tests for simplelearn.nodes
'''

import numpy
import theano
from numpy.testing import assert_allclose
from nose.tools import assert_is_instance, assert_equal
from simplelearn.nodes import (Node,
                               InputNode,
                               Linear,
                               Bias,
                               Function1dTo1d)
from simplelearn.formats import DenseFormat

from unittest import TestCase

import pdb


class DummyFunction1dTo1d(Function1dTo1d):
    def __init__(self, output_format, input_node):
        super(DummyFunction1dTo1d, self).__init__(output_format, input_node)

    def _get_function_of_rows(self, input_rows):
        return input_rows


class Function1dTo1dTester(TestCase):
    def setUp(self):
        input_format = DenseFormat(axes=('0', 'b', '1'),
                                   shape=(3, -1, 4),
                                   dtype=theano.config.floatX)

        self.input_node = InputNode(input_format)

        output_format = DenseFormat(axes=('k', 'b', 'f'),
                                    shape=(6, -1, 2),
                                    dtype=None)

        self.node = self._make_node(output_format, self.input_node)
        assert_is_instance(self.node, Node)

    def _make_node(self, output_format, input_node):
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
            input_batch = input_format.make_batch(is_symbolic=False,
                                                  batch_size=batch_size)
            input_batch[...] = rng.uniform(size=input_batch.shape)

            output_batch = node_function(input_batch)
            expected_output_batch = self.expected_function(input_batch)

            assert_allclose(output_batch, expected_output_batch)


# def test_function1dto1d():
#     input_format = DenseFormat(axes=('0', 'b', '1'),
#                                shape=(3, -1, 4),
#                                dtype=theano.config.floatX)

#     input_node = InputNode(input_format)

#     output_format = DenseFormat(axes=('k', 'b', 'f'),
#                                 shape=(6, -1, 2),
#                                 dtype=None)

#     node = DummyFunction1dTo1d(output_format, input_node)

#     def expected_function(input_batch):
#         input_batch = input_batch.transpose([input_format.axes.index(a)
#                                              for a in ('b', '0', '1')])
#         input_mat = input_batch.reshape((input_batch.shape[0],
#                                          numpy.prod(input_batch.shape[1:])))

#         # output_mat = numpy.dot(input_mat, linear.params.get_value())
#         output_mat = input_mat

#         non_b_axes = [a for a in output_format.axes if a != 'b']
#         non_b_shape = [output_format.shape[output_format.axes.index(a)]
#                        for a in non_b_axes]

#         output_batch = output_mat.reshape([-1] + non_b_shape)
#         output_axes = tuple(['b'] + non_b_axes)
#         output_batch = output_batch.transpose([output_axes.index(a)
#                                                for a in output_format.axes])

#         return output_batch


#     rng = numpy.random.RandomState(14141)

#     node_function = theano.function([input_node.output_symbol],
#                                     node.output_symbol)

#     batch_size = 5

#     for _ in range(3):
#         input_batch = input_format.make_batch(is_symbolic=False,
#                                               batch_size=batch_size)
#         input_batch[...] = rng.uniform(size=input_batch.shape)

#         output_batch = node_function(input_batch)
#         expected_output_batch = expected_function(input_batch)

#         assert_allclose(output_batch, expected_output_batch)

class LinearTester(Function1dTo1dTester):

    def _make_node(self, output_format, input_node):
        return Linear(output_format, input_node)

    def expected_function1dTo1d(self, rows):
        return numpy.dot(rows, self.node.params.get_value())


class BiasTester(Function1dTo1dTester):
    def _make_node(self, output_format, input_node):
        return Bias(output_format, input_node)

    def expected_function1dTo1d(self, rows):
        params = self.node.params.get_value()
        assert_equal(params.shape[0], 1)
        assert_equal(rows.shape[1], params.shape[1])

        return rows + self.node.params.get_value()

def test_l2loss():
    rng = numpy.random.RandomState(3523)

    def expected_loss_function(arg0, arg1):
        diff = arg0 - arg1
        return (diff * diff).sum()

    def make_loss_function(vec_size):
        fmt = DenseFormat(axes=('b', 'f'),
                          shape=(-1, vec_size),
                          dtype=theano.config.floatX)

        input_node_a = InputNode(fmt)
        input_node_b = InputNode(fmt)

        diff = input_node_a.output_symbol - input_node_b.output_symbol
        loss = (diff * diff).sum()

        return theano.function([input_node_a.output_symbol,
                                input_node_b.output_symbol],
                               loss)


    vec_size = 10
    loss_function = make_loss_function(vec_size)

    cast_to_floatX = numpy.cast[theano.config.floatX]

    for batch_size in xrange(4):
        arg0 = cast_to_floatX(rng.uniform(size=(batch_size, vec_size)))
        arg1 = cast_to_floatX(rng.uniform(size=(batch_size, vec_size)))

        expected_loss = expected_loss_function(arg0, arg1)
        loss = loss_function(arg0, arg1)

        assert_allclose(loss, expected_loss)

# def test_linear():

#     input_format = DenseFormat(axes=('0', 'b', '1'),
#                                shape=(3, -1, 4),
#                                dtype=theano.config.floatX)

#     input_node = InputNode(input_format)

#     output_format = DenseFormat(axes=('k', 'b', 'f'),
#                                 shape=(6, -1, 2),
#                                 dtype=None)

#     linear = Linear(output_format, input_node)

#     def expected_function(input_batch):
#         input_batch = input_batch.transpose([input_format.axes.index(a)
#                                              for a in ('b', '0', '1')])
#         input_mat = input_batch.reshape((input_batch.shape[0],
#                                          numpy.prod(input_batch.shape[1:])))

#         output_mat = numpy.dot(input_mat, linear.params.get_value())

#         non_b_axes = [a for a in output_format.axes if a != 'b']
#         non_b_shape = [output_format.shape[output_format.axes.index(a)]
#                        for a in non_b_axes]

#         output_batch = output_mat.reshape([-1] + non_b_shape)
#         output_axes = tuple(['b'] + non_b_axes)
#         output_batch = output_batch.transpose([output_axes.index(a)
#                                                for a in output_format.axes])

#         return output_batch

#     rng = numpy.random.RandomState(14141)
#     param_shape = linear.params.get_value().shape

#     node_function = theano.function([input_node.output_symbol],
#                                     linear.output_symbol)

#     batch_size = 5

#     for _ in range(3):
#         input_batch = input_format.make_batch(is_symbolic=False,
#                                               batch_size=batch_size)
#         input_batch[...] = rng.uniform(size=input_batch.shape)

#         linear.params.set_value(rng.uniform(size=param_shape))

#         output_batch = node_function(input_batch)
#         expected_output_batch = expected_function(input_batch)

#         assert_allclose(output_batch, expected_output_batch)