#! /usr/bin/env python

import theano
import numpy
from theano.tensor import TensorType
from collections import OrderedDict
from simplelearn.nodes import InputNode, AffineTransform, L2Loss, Bias
from simplelearn.formats import DenseFormat
from simplelearn.utils import safe_izip
from nose.tools import assert_equal
import pdb


def main():
    floatX = theano.config.floatX

    feature_size = 2

    input_type = TensorType(dtype=floatX, broadcastable=[False, False])
    input = input_type.make_variable('input')

    bias = theano.shared(numpy.zeros((1, feature_size), dtype=floatX),
                         broadcastable=[True, False])
    output = input + bias

    label = input_type.make_variable('label')

    diff = output - label
    l2loss = (diff * diff).sum()

    grad = theano.gradient.grad
    bias_grads = grad(l2loss, bias)

    # SGD update with momentum

    learning_rate = .01
    momentum = .5

    velocity = theano.shared(numpy.asarray(0.0 * bias.get_value(),
                                           dtype=floatX),
                             name="velocity")

    new_velocity = momentum * velocity - learning_rate * bias_grads

    updates = OrderedDict(((bias, bias + new_velocity),
                           (velocity, new_velocity)))

    training_func = theano.function([input, label],
                                    l2loss,
                                    updates=updates)

    # Problem doesn't occur if we use these simpler updates:
 #    ((affine_node.linear_node.params,
#                                               -linear_grads * learning_rate),
#                                              (affine_node.bias_node.params,
#                                               -bias_grads * learning_rate)))

    cast = numpy.cast[floatX]

    true_bias = cast(numpy.random.uniform())

    num_examples = 99
    batch_size = 3
    assert_equal(num_examples % batch_size, 0)

    input_vectors = cast(numpy.random.uniform(size=(num_examples,
                                                    feature_size)))
    labels = input_vectors + true_bias

    input_batches = input_vectors.reshape((num_examples / batch_size,
                                           batch_size) +
                                          input_vectors.shape[1:])

    label_batches = labels.reshape((num_examples / batch_size,
                                    batch_size) +
                                   labels.shape[1:])

    for input_batch, label_batch in safe_izip(input_batches, label_batches):
        print training_func(input_batch, label_batch)

if __name__ == '__main__':
    main()
