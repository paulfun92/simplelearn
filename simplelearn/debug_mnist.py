import h5py
import theano
from theano.sandbox.cuda import CudaNdarray
import numpy
from nose.tools import assert_equal
from simplelearn.utils import safe_izip, assert_all_is_instance

import pdb

batch_num = 0
mnist_weights = None
mnist_biases = None

# def save_mnist_params(weights, biases, path, prefix=""):
#     assert_equal(len(weights), 3)
#     assert_equal(len(biases), 3)
#     assert_all_is_instance(weights, theano.gof.Variable)
#     assert_all_is_instance(biases, theano.gof.Variable)

#     def create_dataset(name, shared_variable, h5_file):
#         result = h5_file.create_dataset(name,
#                                         data=shared_variable.get_value())

#     with h5py.File(path, mode='w-') as h5_file:
#         for layer_index, (weight, bias) in enumerate(safe_izip(weights,
#                                                                biases)):
#             if layer_index < 2:
#                 assert_equal(weight.ndim, 4)
#             else:
#                 assert_equal(weight.ndim, 2)

#             assert_equal(bias.ndim, 1)
#             h5_file.create_dataset("{}weight_{}".format(prefix, layer_index),
#                                    data=weight.get_value())

#             h5_file.create_dataset("{}bias_{}".format(prefix, layer_index),
#                                    data=bias.get_value())


def _save_mnist_params_impl(weights, biases, path, prefix):
    assert_equal(len(weights), 3)
    assert_equal(len(biases), 3)
    assert_all_is_instance(weights, numpy.ndarray)
    assert_all_is_instance(biases, numpy.ndarray)

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
            h5_file.create_dataset("{}weight_{}".format(prefix, layer_index),
                                   data=weight)

            h5_file.create_dataset("{}bias_{}".format(prefix, layer_index),
                                   data=bias)


def save_mnist_params(weights, biases, path):
    assert_all_is_instance(weights, theano.gof.Variable)
    assert_all_is_instance(biases, theano.gof.Variable)
    weights = [w.get_value() for w in weights]
    biases = [b.get_value() for b in biases]

    _save_mnist_params_impl(weights, biases, path, prefix="")


def save_mnist_grads(weight_grads, bias_grads, path):
    assert_all_is_instance(weight_grads, CudaNdarray)
    assert_all_is_instance(bias_grads, CudaNdarray)
    weight_grads = [numpy.asarray(w) for w in weight_grads]
    bias_grads = [numpy.asarray(b) for b in bias_grads]
    _save_mnist_params_impl(weight_grads, bias_grads, path, prefix="grad_")


def load_mnist_params(h5_path, prefix=""):

    h5_weights = []
    h5_biases = []

    try:
        with h5py.File(h5_path, mode='r') as h5_file:

            for layer_index in range(3):
                h5_weights.append(numpy.asarray(h5_file["{}weight_{}".format(
                    prefix, layer_index)]))

                h5_biases.append(numpy.asarray(h5_file["{}bias_{}".format(
                    prefix, layer_index)]))
    except IOError:
        print("Couldn't open file {}.".format(h5_path))
        raise

    return h5_weights, h5_biases

def load_mnist_grads(h5_path):
    return load_mnist_params(h5_path, prefix="grad_")

def save_mnist_batch(images, labels, h5_path):
    with h5py.File(h5_path, mode='w-') as h5_file:
        h5_file.create_dataset('images', data=images)
        h5_file.create_dataset('labels', data=labels)

def load_mnist_batch(h5_path):
    print("opening batch {}".format(h5_path))
    result = []
    try:
        with h5py.File(h5_path, mode='r') as h5_file:
            result.append(numpy.asarray(h5_file['images']))
            result.append(numpy.asarray(h5_file['labels']))
    except IOError:
        print("Couldn't open {}.".format(h5_path))
        raise

    return tuple(result)
