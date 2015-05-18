'''
nosetests for ../hdf5.py.

These tests require MNIST to be installed in:
  <simplelearn.data.data_path>/mnist/original_files/
'''

import os
import cPickle
import numpy
from nose.tools import (assert_equal, assert_less)
from numpy.testing import assert_array_equal
from simplelearn.utils import safe_izip
from simplelearn.data.mnist import load_mnist


def _file_size_in_bytes(file_path):
    '''
    Returns the file size.
    '''

    return os.stat(file_path).st_size


def test_pickle_h5_dataset():
    '''
    Tests pickling and unpickling of Hdf5Data.
    '''

    # Path for the pickle file, not the .h5 file.
    file_path = '/tmp/test_mnist_test_pickle_hdf5_data.pkl'

    def make_pickle(file_path):
        '''
        Pickles the MNIST dataset.
        '''
        hdf5_data = load_mnist()
        with open(file_path, 'wb') as pickle_file:
            cPickle.dump(hdf5_data,
                         pickle_file,
                         protocol=cPickle.HIGHEST_PROTOCOL)

    make_pickle(file_path)
    assert_less(_file_size_in_bytes(file_path), 1024 * 5)

    def load_pickle(file_path):
        '''
        Loads the MNIST dataset pickled above.
        '''
        with open(file_path, 'rb') as pickle_file:
            return cPickle.load(pickle_file)

    mnist_datasets = load_mnist()
    pickled_mnist_datasets = load_pickle(file_path)

    for (mnist_dataset,
         pickled_mnist_dataset) in safe_izip(mnist_datasets,
                                             pickled_mnist_datasets):
        for (name,
             expected_name,
             fmt,
             expected_fmt,
             tensor,
             expected_tensor) in safe_izip(pickled_mnist_dataset.names,
                                           mnist_dataset.names,
                                           pickled_mnist_dataset.formats,
                                           mnist_dataset.formats,
                                           pickled_mnist_dataset.tensors,
                                           mnist_dataset.tensors):
            assert_equal(name, expected_name)
            assert_equal(fmt, expected_fmt)
            assert_array_equal(tensor, expected_tensor)


# def test_pickle_hdf5_dataset():
#     '''
#     Tests pickling and unpickling of Hdf5Dataset.
#     '''

#     batch_slice = slice(3, 20)
#     file_path = '/tmp/test_mnist_test_pickle_hdf5_dataset.pkl'

#     def make_pickle(batch_slice, file_path):
#         '''
#         Pickles a slice of the MNIST dataset to file_path.
#         '''
#         mnist = load_mnist()

#         mini_mnist = mnist[batch_slice]

#         tensors = tuple(numpy.asarray(t) for t in mini_mnist.tensors)
#         names = mini_mnist.names
#         formats = mini_mnist.formats

#         with open(file_path, 'wb') as pickle_file:
#             cPickle.dump(mini_mnist,
#                          pickle_file,
#                          protocol=cPickle.HIGHEST_PROTOCOL)

#         return (tensors, names, formats)

#     (expected_tensors,
#      expected_names,
#      expected_formats) = make_pickle(batch_slice, file_path)

#     assert_equal(len(expected_tensors), 2)
#     assert_equal(len(expected_names), 2)
#     assert_equal(len(expected_formats), 2)
#     assert_equal(expected_tensors[0].shape[0],
#                  batch_slice.stop - batch_slice.start)
#     assert_equal(expected_tensors[1].shape[0],
#                  batch_slice.stop - batch_slice.start)

#     # Check that we haven't serialized all of MNIST, or even
#     # just the rows in the batch_slice
#     assert_less(_file_size_in_bytes(file_path), 1024 * 20)

#     def load_pickle(file_path):
#         '''
#         Unpickles the Hdf5Dataset pickled by make_pickle above.
#         '''

#         with open(file_path, 'rb') as pickle_file:
#             return cPickle.load(pickle_file)

#     hdf5_dataset = load_pickle(file_path)

#     for (tensor,
#          expected_tensor,
#          name,
#          expected_name,
#          fmt,
#          expected_fmt) in safe_izip(hdf5_dataset.tensors,
#                                     expected_tensors,
#                                     hdf5_dataset.names,
#                                     expected_names,
#                                     hdf5_dataset.formats,
#                                     expected_formats):
#         assert_array_equal(tensor, expected_tensor)
#         assert_equal(name, expected_name)
#         assert_equal(fmt, expected_fmt)
