'''
nosetests for ../h5_dataset.py.

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
    Tests pickling and unpickling of H5Dataset.
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
