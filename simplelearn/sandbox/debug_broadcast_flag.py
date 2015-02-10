#! /usr/bin/env python

import theano
import numpy
from theano.tensor import TensorType
from collections import OrderedDict
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
                             broadcastable=bias_grads.broadcastable,
                             name="velocity")

    new_velocity = momentum * velocity - learning_rate * bias_grads

    use_momentum = True  # Set to False to side-step Theano error

    if use_momentum:
        updates = OrderedDict(((bias, bias + new_velocity),
                               (velocity, new_velocity)))
    else:
        updates = OrderedDict(((bias, bias + (-learning_rate * bias_grads)),))

    training_func = theano.function([input, label],
                                    l2loss,
                                    updates=updates)

    cast = numpy.cast[floatX]

    true_bias = cast(numpy.random.uniform())

    num_examples = 99
    batch_size = 3
    assert num_examples % batch_size == 0

    input_vectors = cast(numpy.random.uniform(size=(num_examples,
                                                    feature_size)))
    labels = input_vectors + true_bias

    input_batches = input_vectors.reshape((num_examples / batch_size,
                                           batch_size) +
                                          input_vectors.shape[1:])

    label_batches = labels.reshape((num_examples / batch_size,
                                    batch_size) +
                                   labels.shape[1:])

    for input_batch, label_batch in zip(input_batches, label_batches):
        print training_func(input_batch, label_batch)

if __name__ == '__main__':
    main()
