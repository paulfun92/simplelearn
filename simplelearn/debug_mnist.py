import h5py
import theano
import numpy
from nose.tools import assert_equal
from simplelearn.utils import safe_izip, assert_all_is_instance

import pdb

batch_num = 0
mnist_weights = None
mnist_biases = None

def save_mnist_params(weights, biases, path):
    assert_equal(len(weights), 3)
    assert_equal(len(biases), 3)
    assert_all_is_instance(weights, theano.gof.Variable)
    assert_all_is_instance(biases, theano.gof.Variable)

    def create_dataset(name, shared_variable, h5_file):
        result = h5_file.create_dataset(name,
                                        data=shared_variable.get_value())

    with h5py.File(path, mode='w-') as h5_file:
        for layer_index, (weight, bias) in enumerate(safe_izip(weights,
                                                               biases)):
            if layer_index < 2:
                assert_equal(weight.ndim, 4)
            else:
                assert_equal(weight.ndim, 2)

            assert_equal(bias.ndim, 1)
            h5_file.create_dataset("weight_{}".format(layer_index),
                                   data=weight.get_value())

            h5_file.create_dataset("bias_{}".format(layer_index),
                                   data=bias.get_value())



def load_mnist_params(h5_path):

    h5_weights = []
    h5_biases = []

    with h5py.File(h5_path, mode='r') as h5_file:

        for layer_index in range(3):
            h5_weights.append(numpy.asarray(h5_file["weight_{}".format(layer_index)]))
            h5_biases.append(numpy.asarray(h5_file["bias_{}".format(layer_index)]))

    return h5_weights, h5_biases


def save_mnist_batch(images, labels, h5_path):
    with h5py.File(h5_path, mode='w-') as h5_file:
        h5_file.create_dataset('images', data=images)
        h5_file.create_dataset('labels', data=labels)

def load_mnist_batch(h5_path):
    result = []
    with h5py.File(h5_path, mode='r') as h5_file:
        result.append(numpy.asarray(h5_file['images']))
        result.append(numpy.asarray(h5_file['labels']))

    return tuple(result)
